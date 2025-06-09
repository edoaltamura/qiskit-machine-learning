# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Gradient of probabilities with parameter shift
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import sys

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSamplerV1
from qiskit.primitives.base import BaseSamplerV2
from qiskit.result import QuasiDistribution

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult
from ..utils import _make_param_shift_parameter_values
from ...exceptions import AlgorithmError


class ParamShiftSamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the parameter shift rule [1].

    **Reference:**
    [1] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., and Killoran, N. Evaluating analytic
    gradients on quantum hardware, `DOI <https://doi.org/10.1103/PhysRevA.99.032331>`_
    """

    SUPPORTED_GATES = [
        "x",
        "y",
        "z",
        "h",
        "rx",
        "ry",
        "rz",
        "p",
        "cx",
        "cy",
        "cz",
        "ryy",
        "rxx",
        "rzz",
        "rzx",
    ]

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the estimator gradients on the given circuits."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(g_circuits, g_parameter_values, g_parameters, **options)
        return self._postprocess(results, circuits, parameter_values, parameters)



    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute sampler gradients on the fly, printing array sizes in MB."""
        # prepare jobs and mapping to circuits
        job_circuits, job_param_values, metadata = [], [], []
        all_n = []
        for idx, (circuit, param_vals, params_) in enumerate(zip(circuits, parameter_values, parameters)):
            metadata.append({"parameters": params_})
            shifted_vals = _make_param_shift_parameter_values(circuit, param_vals, params_)
            n = len(shifted_vals)
            job_circuits.extend([circuit] * n)
            job_param_values.extend(shifted_vals)
            all_n.append(n)

        # build index-to-circuit map
        idx_to_circ = []
        for circ_idx, n in enumerate(all_n):
            idx_to_circ.extend([circ_idx] * n)

        # initialize buffers for each circuit
        buffers = [[] for _ in circuits]
        gradients = [None] * len(circuits)
        opt = None

        batch_size = 10
        total = len(job_circuits)
        num_batches = (total + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_circs = job_circuits[start:end]
            batch_params = job_param_values[start:end]

            # run sampler for this batch
            if isinstance(self._sampler, BaseSamplerV1):
                job = self._sampler.run(batch_circs, batch_params, **options)
                batch_res = job.result().quasi_dists
                opt = self._get_local_options(options)

            elif isinstance(self._sampler, BaseSamplerV2):
                if self._pass_manager is None:
                    circs = batch_circs
                    len_qd = 2 ** batch_circs[0].num_qubits
                else:
                    circs = self._pass_manager.run(batch_circs)
                    len_qd = 2 ** circs[0].layout._input_qubit_count
                circ_params = [(circs[i], batch_params[i]) for i in range(len(batch_params))]
                job = self._sampler.run(circ_params)
                batch_res = job.result()
                opt = options

            else:
                raise AlgorithmError(
                    "Accepted samplers are BaseSamplerV1 or BaseSamplerV2; got "
                    f"{type(self._sampler)}."
                )

            # process each result in batch
            for local_idx, res in enumerate(batch_res):
                global_idx = start + local_idx
                circ_idx = idx_to_circ[global_idx]

                if isinstance(self._sampler, BaseSamplerV1):
                    # res is already a quasi-distribution
                    buffers[circ_idx].append(res)

                else:  # BaseSamplerV2
                    # extract counts
                    if hasattr(res.data, "meas"):
                        counts = res.data.meas.get_counts()
                    else:
                        counts = res.data.c.get_counts()
                    total_shots = sum(counts.values())
                    probs = {k: v / total_shots for k, v in counts.items()}
                    qdist = QuasiDistribution(probs)
                    filtered = {k: v for k, v in qdist.items() if int(k) < len_qd}
                    buffers[circ_idx].append(filtered)

                # once we have all shifted results for this circuit, compute gradient
                if len(buffers[circ_idx]) == all_n[circ_idx]:
                    dist_list = buffers[circ_idx]
                    grads = []
                    half = all_n[circ_idx] // 2
                    for plus, minus in zip(dist_list[:half], dist_list[half:]):
                        grad_dist = defaultdict(float)
                        for key, val in plus.items():
                            grad_dist[key] += val / 2
                        for key, val in minus.items():
                            grad_dist[key] -= val / 2
                        grads.append(dict(grad_dist))
                    gradients[circ_idx] = grads
                    # free buffer
                    buffers[circ_idx] = []


        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)
