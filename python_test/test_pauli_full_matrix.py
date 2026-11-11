import numpy as np
import pytest
import scaluq

PAULI_MATRICES = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _embedded_pauli_matrix(name, target, n_qubits):
    factors = [np.eye(2, dtype=np.complex128) for _ in range(n_qubits)]
    factors[n_qubits - 1 - target] = PAULI_MATRICES[name]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("coef", [0, 2, 0.5 - 0.25j])
def test_pauli_full_matrix_includes_coefficient(precision, coef):
    if not scaluq.precision_available(precision):
        pytest.skip(f"{precision} precision is unavailable")
    module = getattr(scaluq.default, precision)
    tolerance = 1e-6 if precision == "f32" else 1e-12

    identity = module.PauliOperator("", coef)
    np.testing.assert_allclose(
        identity.get_full_matrix(0), np.array([[coef]]), atol=tolerance
    )
    np.testing.assert_allclose(
        identity.get_full_matrix_ignoring_coef(0), [[1]], atol=tolerance
    )

    for name in ("X", "Y", "Z"):
        pauli = module.PauliOperator(f"{name} 1", coef)
        expected = _embedded_pauli_matrix(name, target=1, n_qubits=3)
        np.testing.assert_allclose(
            pauli.get_full_matrix(3), coef * expected, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            pauli.get_full_matrix_ignoring_coef(3),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )

        single_qubit = module.PauliOperator(f"{name} 0", coef)
        np.testing.assert_allclose(
            single_qubit.get_full_matrix(1),
            single_qubit.get_matrix(),
            rtol=tolerance,
            atol=tolerance,
        )
