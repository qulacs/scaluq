import numpy as np
import pytest
import scaluq


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("space", ["default", "host_serial"])
def test_z_sum_diagonal_product(precision, space):
    state = scaluq.StateVector(3, precision=precision, space=space)
    diagonal = scaluq.StateVector(3, precision=precision, space=space)
    data = np.array([1, 2j, 3 - 1j, -2, 0, 1 + 1j, -1j, 2], dtype=complex)
    basis = np.arange(8)
    # (0.3 + 0.2j) I + 0.5 Z0 - 0.1j Z1 Z2.
    weights = 0.3 + 0.2j + 0.5 * (1 - 2 * (basis & 1))
    weights -= 0.1j * (1 - 2 * (((basis >> 1) ^ (basis >> 2)) & 1))
    state.load(data)
    diagonal.load(weights)
    state.multiply_elementwise(diagonal)
    tolerance = 1e-6 if precision == "f32" else 1e-12
    np.testing.assert_allclose(state.get_amplitudes(), data * weights, atol=tolerance)
    np.testing.assert_allclose(diagonal.get_amplitudes(), weights, atol=tolerance)
    state.multiply_elementwise(state)
    np.testing.assert_allclose(state.get_amplitudes(), (data * weights)**2, atol=tolerance)
    with pytest.raises(ValueError):
        state.multiply_elementwise(scaluq.StateVector(2, precision=precision, space=space))
