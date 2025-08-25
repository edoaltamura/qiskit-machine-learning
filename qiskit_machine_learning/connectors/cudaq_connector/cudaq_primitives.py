# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The CUDA-Q connector primitives."""

import cudaq
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
from qiskit.primitives import EstimatorResult, SamplerResult
from qiskit.result import QuasiDistribution

from .cudaq_connector import translate_circuit


class CudaQEstimator(BaseEstimatorV2):
    """Qiskit Estimator V2 wrapper around CUDA-Q observe()."""

    def __init__(self, shots: int = 1024):
        super().__init__()
        self.shots = shots

    def run(self, publist, execution=None, **kwargs):
        """publist is a list of (circuit, observable, parameter_values)."""
        _require_cudaq()
        values, metadata = [], []
        for circuit, observable, params in publist:
            spec = translate_circuit(circuit)
            if len(params) != len(spec.param_order):
                raise ValueError(f"Expected {len(spec.param_order)} parameter values, got {len(params)}.")
            res = cudaq.observe(spec.kernel, observable, *params, execution=execution)
            values.append(res.expectation())
            metadata.append({"shots": self.shots})
        return _CudaQJob(EstimatorResult(values=values, metadata=metadata))


class CudaQSampler(BaseSamplerV2):
    """Qiskit Sampler V2 wrapper around CUDA-Q sample()."""

    def __init__(self, shots: int = 1024):
        super().__init__()
        self.shots = shots

    def run(self, publist, execution=None, **kwargs):
        """publist is a list of (circuit, parameter_values)."""
        _require_cudaq()
        quasi_dists, metadata = [], []
        for circuit, params in publist:
            spec = translate_circuit(circuit)
            if len(params) != len(spec.param_order):
                raise ValueError(f"Expected {len(spec.param_order)} parameter values, got {len(params)}.")
            counts = cudaq.sample(spec.kernel, *params, shots_count=self.shots, execution=execution)
            qdist = {k: v / self.shots for k, v in counts.items()}
            quasi_dists.append(QuasiDistribution(qdist))
            metadata.append({"shots": self.shots})

        return _CudaQJob(SamplerResult(quasi_dists=quasi_dists, metadata=metadata))


class CudaQAsyncEstimator(BaseEstimatorV2):
    def __init__(self, shots: int = 1024):
        super().__init__()
        self.shots = shots
        self.qpu_count = cudaq.get_target().num_qpus()

    def run(self, publist, **kwargs):
        _require_cudaq()
        futures = []
        for i, (circuit, observable, params) in enumerate(publist):
            spec = translate_circuit(circuit)
            if len(params) != len(spec.param_order):
                raise ValueError(f"Expected {len(spec.param_order)} parameter values, got {len(params)}.")
            qpu_id = i // self.qpu_count
            fut = cudaq.observe_async(spec.kernel, observable, *params,
                                      shots_count=self.shots, qpu_id=qpu_id)
            futures.append(fut)

        # Go do other work, asynchronous execution of sample tasks ongoing.
        # Get the results, note future::get() will kick off a wait
        # if the results are not yet available.
        values, metadata = [], []
        for idx in range(len(futures)):
            values.append(futures[idx].get().expectation())
            metadata.append({"shots": self.shots})

        return _CudaQJob(EstimatorResult(values=values, metadata=metadata))


class CudaQAsyncSampler(BaseSamplerV2):
    def __init__(self, shots: int = 1024):
        super().__init__()
        self.shots = shots
        self.qpu_count = cudaq.get_target().num_qpus()

    def run(self, publist, **kwargs):
        _require_cudaq()
        futures = []
        for i, (circuit, params) in enumerate(publist):
            spec = translate_circuit(circuit)
            if len(params) != len(spec.param_order):
                raise ValueError(
                    f"Expected {len(spec.param_order)} parameter values, got {len(params)}.")
            qpu_id = i // self.qpu_count
            fut = cudaq.sample_async(spec.kernel, *params,
                                     shots_count=self.shots, qpu_id=qpu_id)
            futures.append(fut)

        # Go do other work, asynchronous execution of sample tasks on-going.
        # Get the results, note future::get() will kick off a wait
        # if the results are not yet available.
        quasi_dists, metadata = [], []
        for idx in range(len(futures)):
            counts = futures[idx].get()
            qdist = {k: v / self.shots for k, v in counts.items()}
            quasi_dists.append(QuasiDistribution(qdist))
            metadata.append({"shots": self.shots})

        return _CudaQJob(SamplerResult(quasi_dists=quasi_dists, metadata=metadata))


class _CudaQJob:
    def __init__(self, result_obj):
        self._result = result_obj

    def result(self):
        return self._result


def _require_cudaq():
    if cudaq is None:
        raise RuntimeError("CUDA-Q is not available. Run `pip install cudaq` and make sure it imports.")
