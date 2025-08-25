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
"""Tests of the CUDA-Q connector primitives."""

import pytest
import numpy as np

import cudaq
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Pauli
from qiskit.primitives import StatevectorEstimator as QiskitEstimator, StatevectorSampler as QiskitSampler

from cudaq_connector import (
    CudaQEstimator, CudaQSampler, CudaQAsyncEstimator, CudaQAsyncSampler
)


# ----------------------------
# Helpers
# ----------------------------

def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def compare_quasi_distributions(qd1, qd2, atol=1e-6):
    """Compare two QuasiDistributions (dict-like)."""
    keys = set(qd1.keys()) | set(qd2.keys())
    for k in keys:
        v1, v2 = qd1.get(k, 0.0), qd2.get(k, 0.0)
        assert np.isclose(v1, v2, atol=atol), f"Mismatch at {k}: {v1} vs {v2}"


def compare_expectations(v1, v2, atol=1e-6):
    assert np.isclose(v1, v2, atol=atol), f"Mismatch in expectation: {v1} vs {v2}"


# ----------------------------
# Tests: Estimator
# ----------------------------

@pytest.mark.parametrize("EstimatorClass", [CudaQEstimator, CudaQAsyncEstimator])
def test_estimator_bell(EstimatorClass):
    qc = bell_circuit()
    obs = Pauli("ZZ")

    ref_est = QiskitEstimator().run([(qc, obs)]).result()
    cudaq_est = EstimatorClass().run([(qc, obs, [])]).result()

    compare_expectations(ref_est.values[0], cudaq_est.values[0])


@pytest.mark.parametrize("EstimatorClass", [CudaQEstimator, CudaQAsyncEstimator])
def test_estimator_random_circuits(EstimatorClass):
    rng = np.random.default_rng(123)
    for n in [2, 3]:
        qc = random_circuit(n, depth=3, seed=rng)
        obs = Pauli("Z" * n)  # test expectation of ZZZ...Z

        ref_est = QiskitEstimator().run([(qc, obs)]).result()
        cudaq_est = EstimatorClass().run([(qc, obs, [])]).result()

        compare_expectations(ref_est.values[0], cudaq_est.values[0])


# ----------------------------
# Tests: Sampler
# ----------------------------

@pytest.mark.parametrize("SamplerClass", [CudaQSampler, CudaQAsyncSampler])
def test_sampler_bell(SamplerClass):
    qc = bell_circuit()

    ref_samp = QiskitSampler(shots=1024).run([(qc, [])]).result()
    cudaq_samp = SamplerClass(shots=1024).run([(qc, [])]).result()

    compare_quasi_distributions(
        ref_samp.quasi_dists[0].dict(),
        cudaq_samp.quasi_dists[0].dict(),
        atol=0.05,  # allow sampling noise
    )


@pytest.mark.parametrize("SamplerClass", [CudaQSampler, CudaQAsyncSampler])
def test_sampler_random_circuits(SamplerClass):
    rng = np.random.default_rng(321)
    for n in [2, 3]:
        qc = random_circuit(n, depth=3, seed=rng)

        ref_samp = QiskitSampler(shots=2048).run([(qc, [])]).result()
        cudaq_samp = SamplerClass(shots=2048).run([(qc, [])]).result()

        compare_quasi_distributions(
            ref_samp.quasi_dists[0].dict(),
            cudaq_samp.quasi_dists[0].dict(),
            atol=0.05,
        )
