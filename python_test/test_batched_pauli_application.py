import numpy as np
import pytest
import scaluq


@pytest.fixture(params=["f32", "f64"])
def precision(request):
    if not scaluq.precision_available(request.param):
        pytest.skip(f"{request.param} unavailable")
    return request.param


@pytest.fixture(params=["default", "host", "host_serial"])
def space(request):
    if request.param == "host" and not hasattr(scaluq, "host"):
        pytest.skip("host module unavailable")
    return request.param


@pytest.mark.parametrize("pauli", ["", "X 0", "Y 0", "Z 1", "Y 0 X 1"])
def test_batched_pauli_matches_dense_matrix(space, precision, pauli):
    values = np.array([[1, 1j, 2 - 1j, -.5 + .25j],
                       [-1 + 1j, 2, -2j, 1 + .5j],
                       [.25 + .5j, -.5, 1 - 1j, 3 + 2j]], dtype=complex)
    states = scaluq.StateVectorBatched(3, 2, precision=precision, space=space)
    states.load(values.tolist())
    op = scaluq.PauliOperator(pauli, coef=.5 - .25j, precision=precision)
    op.apply_to_state(states)
    np.testing.assert_allclose(states.get_amplitudes(), values @ np.asarray(op.get_full_matrix_ignoring_coef(2)).T * op.coef())


def test_batched_pauli_scalar_empty_and_invalid(space, precision):
    states = scaluq.StateVectorBatched(3, 0, precision=precision, space=space)
    states.load([[1], [1j], [2]])
    scaluq.PauliOperator("", 2j, precision=precision).apply_to_state(states)
    np.testing.assert_allclose(states.get_amplitudes(), [[2j], [-2], [4j]])
    with pytest.raises(RuntimeError):
        scaluq.PauliOperator("X 0", precision=precision).apply_to_state(states)
    with pytest.raises(RuntimeError):
        scaluq.PauliOperator("Z 0", precision=precision).apply_to_state(states)
    np.testing.assert_allclose(states.get_amplitudes(), [[2j], [-2], [4j]])
    empty = scaluq.StateVectorBatched(0, 1, precision=precision, space=space)
    scaluq.PauliOperator("Y 0", precision=precision).apply_to_state(empty)
    with pytest.raises(RuntimeError):
        scaluq.PauliOperator("X 1", precision=precision).apply_to_state(empty)
