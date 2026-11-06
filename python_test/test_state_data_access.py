import numpy as np
import pytest

import scaluq


@pytest.mark.parametrize("precision,dtype", [("f32", np.complex64), ("f64", np.complex128)])
def test_numpy_transfer_and_data_access(precision, dtype):
    state = scaluq.StateVector(3, precision=precision)
    data = np.arange(8, dtype=dtype) * (1 + 1j)
    state.load_numpy(data)
    result = state.get_amplitudes_numpy()
    np.testing.assert_array_equal(result, data)
    assert result.dtype == dtype
    result[1] = 100
    assert state.get_amplitude_at(1) == data[1]
    data[1] = 99
    assert state.get_amplitude_at(1) == 1 + 1j
    data[1] = 1 + 1j
    np.testing.assert_array_equal(state.get_amplitudes(2, 3), data[2:5])
    sub = state.extract_substate([1], [1])
    np.testing.assert_array_equal(sub.get_amplitudes_numpy(), data[[2, 3, 6, 7]])
    state.embed_substate(sub, [2, 0])
    np.testing.assert_array_equal(state.get_amplitudes_numpy(), data[[2, 6, 0, 0, 3, 7, 0, 0]])
    with pytest.raises(Exception):
        state.load_numpy(data[::2])
    with pytest.raises(Exception):
        state.load_numpy(np.arange(8))
    with pytest.raises(ValueError):
        state.extract_substate([0, 0], [0, 1])
