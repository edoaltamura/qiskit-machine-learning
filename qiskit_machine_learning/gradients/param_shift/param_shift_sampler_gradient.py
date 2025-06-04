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

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit import generate_preset_pass_manager
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
        """Compute sampler gradients in small batches with finite shots."""
        # build flat list of all shifted circuits
        job_circuits: list[QuantumCircuit] = []
        job_param_values: list[dict[Parameter, float]] = []
        metadata: list[dict[str, Sequence[Parameter]]] = []
        all_n: list[int] = []

        for circ, vals, params in zip(circuits, parameter_values, parameters):
            metadata.append({"parameters": params})
            shifts = _make_param_shift_parameter_values(circ, vals, params)
            n_shifts = len(shifts)
            all_n.append(n_shifts)
            job_circuits.extend([circ] * n_shifts)
            job_param_values.extend(shifts)

        total_shifts = sum(all_n)
        if total_shifts == 0:
            return SamplerGradientResult(
                gradients=[[] for _ in circuits],
                metadata=metadata,
                options=options,
            )

        # force finite shots for BaseSamplerV2
        run_opts = options.copy()
        run_opts.setdefault("shots", 1024)
        batch_size = run_opts.pop("batch_size", 10)

        # pre-transpile each original circuit once
        pm = generate_preset_pass_manager(backend=None)
        transpiled_map: dict[int, QuantumCircuit] = {}
        for idx, orig in enumerate(circuits):
            transpiled_map[idx] = pm.run(orig)

        # prepare sparse gradient storage
        gradients: list[list[dict[int, float]]] = [[] for _ in circuits]

        # stream through shifts in batches
        processed = 0
        partial_sum = 0  # tracks how many shifts from previous circuits
        circ_idx = 0
        shift_accum = 0

        while processed < total_shifts:
            end = min(processed + batch_size, total_shifts)
            sub_circs: list[QuantumCircuit] = []
            sub_vals: list[dict[Parameter, float]] = []

            # bind each shift in this batch to its transpiled circuit
            idx = processed
            temp_circ_idx = circ_idx
            temp_shift_accum = shift_accum
            while idx < end:
                # move forward to correct circuit block
                while temp_shift_accum + all_n[temp_circ_idx] <= idx:
                    temp_shift_accum += all_n[temp_circ_idx]
                    temp_circ_idx += 1
                offset = idx - temp_shift_accum
                base = transpiled_map[temp_circ_idx]
                bind = job_param_values[idx]
                bound = base.assign_parameters(bind, inplace=False)
                sub_circs.append(bound)
                sub_vals.append(bind)
                idx += 1

            # run this batch
            if isinstance(self._sampler, BaseSamplerV1):
                sub_job = self._sampler.run(sub_circs, sub_vals, **run_opts)
                sub_res = sub_job.result()
                sub_quasi = sub_res.quasi_dists
            elif isinstance(self._sampler, BaseSamplerV2):
                if self._pass_manager:
                    sub_circs = self._pass_manager.run(sub_circs)
                n_qubits = sub_circs[0].num_qubits
                cutoff = 2 ** n_qubits
                circ_params = [(sub_circs[i], sub_vals[i]) for i in range(len(sub_circs))]
                sub_job = self._sampler.run(circ_params, **run_opts)
                sub_res = sub_job.result()
                sub_quasi = []
                for r in sub_res:
                    data = r.data
                    counts = data.meas.get_counts() if hasattr(data, "meas") else data.c.get_counts()
                    total = sum(counts.values())
                    probs = {bs: cnt / total for bs, cnt in counts.items()}
                    filtered = {int(bs, 2): p for bs, p in probs.items() if int(bs, 2) < cutoff}
                    sub_quasi.append(filtered)
            else:
                raise AlgorithmError(
                    "Accepted estimators are BaseSamplerV1 and BaseSamplerV2; got "
                    f"{type(self._sampler)}."
                )

            # assemble gradients from this batch
            local_start = processed
            temp_circ_idx = circ_idx
            temp_shift_accum = shift_accum
            idx2 = processed
            for i in range(len(sub_quasi)):
                # find circuit index for this flattened shift
                while temp_shift_accum + all_n[temp_circ_idx] <= idx2:
                    temp_shift_accum += all_n[temp_circ_idx]
                    temp_circ_idx += 1
                offset = idx2 - temp_shift_accum
                half = all_n[temp_circ_idx] // 2
                param_slot = offset % half
                is_minus = offset // half
                grad_dict = gradients[temp_circ_idx]
                if offset == 0:
                    grad_dict.extend([{} for _ in range(half)])
                factor = 0.5 if is_minus == 0 else -0.5
                for key, prob in sub_quasi[i].items():
                    grad_dict[param_slot][key] = grad_dict[param_slot].get(key, 0.0) + factor * prob
                idx2 += 1

            # advance processed counters to next batch
            processed = end
            # update circ_idx and shift_accum
            while circ_idx < len(all_n) and shift_accum + all_n[circ_idx] <= processed:
                shift_accum += all_n[circ_idx]
                circ_idx += 1

            # free batch memory
            del sub_quasi, sub_res, sub_job

        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=options)
