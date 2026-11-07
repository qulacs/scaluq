import numpy as np
import pytest

from scaluq import default


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
@pytest.mark.parametrize("coef", [0.7 + 0j, 0.55 - 0.2j])
@pytest.mark.parametrize("control_value", [0, 1])
def test_single_qubit_pauli_rotation_matches_matrix(pauli, coef, control_value):
    f64 = default.f64
    angle = -0.83
    state = f64.StateVector(2)
    initial = np.array([0.3 + 0.2j, -0.4 + 0.1j, 0.7 - 0.5j, -0.2 - 0.6j])
    state.load(initial.tolist())
    gate = f64.gate.PauliRotation(
        f64.PauliOperator(f"{pauli} 1", coef), angle, [0], [control_value]
    )
    matrix = np.asarray(gate.get_matrix())

    expected = initial.copy()
    for base in (0, 1):
        if base == control_value:
            one = base | 2
            expected[base] = matrix[0, 0] * initial[base] + matrix[0, 1] * initial[one]
            expected[one] = matrix[1, 0] * initial[base] + matrix[1, 1] * initial[one]

    gate.update_quantum_state(state)
    np.testing.assert_allclose(state.get_amplitudes(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("space", ["default", "host_serial"])
@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
@pytest.mark.parametrize("coef", [-0.7 + 0j, 0.55 - 0.2j])
def test_param_pauli_rotation_batch_matches_matrix(precision, space, pauli, coef):
    import scaluq
    module = getattr(default, precision)
    params = [0.0, 0.3, -0.7]
    initial = np.array([[0.3 + 0.2j, -0.4 + 0.1j, 0.7 - 0.5j, -0.2 - 0.6j]])
    initial = np.repeat(initial, len(params), axis=0)
    states = scaluq.StateVectorBatched(len(params), 2, precision=precision, space=space)
    states.load(initial.tolist())
    gate = module.gate.ParamPauliRotation(module.PauliOperator(f"{pauli} 1", coef), 1.4, [0], [0])
    expected = initial.copy()
    for b, param in enumerate(params):
        matrix = np.asarray(gate.get_matrix(param))
        expected[b, [0, 2]] = matrix @ initial[b, [0, 2]]
    gate.update_quantum_state(states, param=params)
    tolerance = 1e-5 if precision == "f32" else 1e-12
    np.testing.assert_allclose(states.get_amplitudes(), expected, rtol=tolerance, atol=tolerance)
