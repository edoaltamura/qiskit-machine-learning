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
"""Tests of the CUDA-Q connector translator."""

import pytest
import numpy as np
import cudaq
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from cudaq_connector.cudaq_connector import translate_circuit


def run_cudaq_kernel(kernel, n_qubits, *params):
    """Helper: run a CUDA-Q kernel and return statevector as np.array."""
    sv = cudaq.get_state(lambda: kernel(*params))
    return np.array(sv, dtype=complex)


def statevector_qiskit(circ, param_bindings=None):
    """Helper: get Qiskit statevector."""
    if param_bindings:
        circ = circ.assign_parameters(param_bindings)
    return np.array(Statevector.from_instruction(circ).data, dtype=complex)


def assert_state_close(sv1, sv2, tol=1e-6):
    """Check statevectors are equal up to global phase."""
    # pick nonzero ref to compute relative global phase
    idx = np.argmax(np.abs(sv2))
    phase = sv1[idx] / sv2[idx] if abs(sv2[idx]) > 1e-12 else 1.0
    assert np.allclose(sv1, phase * sv2, atol=tol)


# -------------------------
# FIXED TESTS
# -------------------------

def test_single_hadamard():
    qc = QuantumCircuit(1)
    qc.h(0)
    spec = translate_circuit(qc)
    cuda_sv = run_cudaq_kernel(spec.kernel, 1)
    qiskit_sv = statevector_qiskit(qc)
    assert_state_close(cuda_sv, qiskit_sv)


def test_two_qubit_bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    spec = translate_circuit(qc)
    cuda_sv = run_cudaq_kernel(spec.kernel, 2)
    qiskit_sv = statevector_qiskit(qc)
    assert_state_close(cuda_sv, qiskit_sv)


def test_parameterized_rotation():
    from qiskit.circuit import Parameter
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    spec = translate_circuit(qc)
    cuda_sv = run_cudaq_kernel(spec.kernel, 1, np.pi/2)
    qiskit_sv = statevector_qiskit(qc, {theta: np.pi/2})
    assert_state_close(cuda_sv, qiskit_sv)


# -------------------------
# RANDOMIZED TESTS
# -------------------------

@pytest.mark.parametrize("n_qubits,depth,seed", [
    (2, 3, 42),
    (3, 4, 123),
    (4, 5, 999),
])
def test_random_circuits(n_qubits, depth, seed):
    qc = random_circuit(num_qubits=n_qubits, depth=depth, seed=seed)
    spec = translate_circuit(qc)

    cuda_sv = run_cudaq_kernel(spec.kernel, n_qubits)
    qiskit_sv = statevector_qiskit(qc)

    assert_state_close(cuda_sv, qiskit_sv)
