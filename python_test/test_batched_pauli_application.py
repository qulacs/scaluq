import numpy as np
import pytest
import scaluq


@pytest.mark.parametrize("space", ["default", "host", "host_serial"])
@pytest.mark.parametrize("pauli", ["", "X 0", "Y 0", "Z 1", "Y 0 X 1"])
def test_batched_pauli_matches_dense_matrix(space, pauli):
    precision = "f64"
    if not scaluq.precision_available(precision):
        pytest.skip("f64 unavailable")
    values = np.array([[1, 1j, 2 - 1j, -.5 + .25j],
                       [-1 + 1j, 2, -2j, 1 + .5j],
                       [.25 + .5j, -.5, 1 - 1j, 3 + 2j]], dtype=complex)
    states = scaluq.StateVectorBatched(3, 2, precision=precision, space=space)
    states.load(values.tolist())
    op = scaluq.PauliOperator(pauli, coef=.5 - .25j, precision=precision)
    op.apply_to_state(states)
    np.testing.assert_allclose(states.get_amplitudes(), values @ np.asarray(op.get_full_matrix_ignoring_coef(2)).T * op.coef())


@pytest.mark.parametrize("space", ["default", "host", "host_serial"])
def test_batched_pauli_scalar_empty_and_invalid(space):
    if not scaluq.precision_available("f64"):
        pytest.skip("f64 unavailable")
    states = scaluq.StateVectorBatched(3, 0, precision="f64", space=space)
    states.load([[1], [1j], [2]])
    scaluq.PauliOperator("", 2j, precision="f64").apply_to_state(states)
    np.testing.assert_allclose(states.get_amplitudes(), [[2j], [-2], [4j]])
    with pytest.raises(RuntimeError):
        scaluq.PauliOperator("X 0", precision="f64").apply_to_state(states)
    with pytest.raises(RuntimeError):
        scaluq.PauliOperator("Z 0", precision="f64").apply_to_state(states)
    np.testing.assert_allclose(states.get_amplitudes(), [[2j], [-2], [4j]])
    empty = scaluq.StateVectorBatched(0, 1, precision="f64", space=space)
    scaluq.PauliOperator("Y 0", precision="f64").apply_to_state(empty)
    with pytest.raises(RuntimeError):
        scaluq.PauliOperator("X 1", precision="f64").apply_to_state(empty)
