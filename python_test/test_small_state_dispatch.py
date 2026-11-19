import numpy as np
import pytest
import scaluq


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_serial_threshold(precision):
    previous = scaluq.get_small_state_serial_threshold()
    try:
        def run(threshold):
            scaluq.set_small_state_serial_threshold(threshold)
            state = scaluq.StateVector.Haar_random_state(5, seed=12, precision=precision)
            scaluq.gate.H(0, precision=precision).update_quantum_state(state)
            scaluq.gate.X(1, [4], [0], precision=precision).update_quantum_state(state)
            return state.get_amplitudes()
        expected = run(0)
        np.testing.assert_allclose(run(32), expected, atol=1e-6)
        assert scaluq.get_small_state_serial_threshold() == 32
        with pytest.raises(TypeError):
            scaluq.set_small_state_serial_threshold(-1)
    finally:
        scaluq.set_small_state_serial_threshold(previous)
