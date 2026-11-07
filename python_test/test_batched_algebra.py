import numpy as np
import pytest
import scaluq


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_weighted_batch(precision):
    batch = scaluq.StateVectorBatched(3, 1, precision=precision)
    x = np.array([[1+2j, 2-1j], [3+1j, -1+2j], [2-2j, 4+0j]])
    weights = [1-1j, .5+.5j, -1j]
    batch.load(x.tolist())
    np.testing.assert_allclose(batch.linear_combination(weights).get_amplitudes(), weights @ x, atol=1e-5)
    scaled = batch.copy()
    scaled.multiply_coef(weights)
    np.testing.assert_allclose(scaled.get_amplitudes(), np.array(weights)[:, None] * x, atol=1e-5)
    np.testing.assert_allclose(scaluq.StateVectorBatched.inner_product(batch, scaled),
                               (x.conj() * (np.array(weights)[:, None] * x)).sum(axis=1), atol=1e-5)
    scaled.add_state_vector_with_coef(weights, batch)
    np.testing.assert_allclose(scaled.get_amplitudes(), 2 * np.array(weights)[:, None] * x, atol=1e-5)
    before = scaled.get_amplitudes()
    with pytest.raises(ValueError):
        scaled.add_state_vector_with_coef([1], batch)
    assert scaled.get_amplitudes() == before
    empty = scaluq.StateVectorBatched(0, 0, precision=precision)
    assert empty.linear_combination([]).get_amplitudes() == [0j]
    assert scaluq.StateVectorBatched.inner_product(empty, empty) == []
    wrong = scaluq.StateVectorBatched(2, 1, precision=precision)
    with pytest.raises(ValueError):
        scaluq.StateVectorBatched.inner_product(batch, wrong)
    with pytest.raises(ValueError):
        batch.linear_combination([1])
