import pytest
import scaluq
from scaluq import default


def test_grouped_cached_and_streaming():
    m = default.f64
    terms = [m.PauliOperator('Y 0', 1 + 2j), m.PauliOperator('X 0 Z 1', 0.5 - 1j)]
    x = m.StateVector(2)
    x.load([1, 2j, 3, 4])
    expected = x.copy()
    expected.set_zero_norm_state()
    for term in terms:
        applied = x.copy()
        term.apply_to_state(applied)
        expected.add_state_vector_with_coef(1, applied)
    for budget in (0, 4096):
        op = scaluq.GroupedPauliOperator(2, terms, budget, precision='f64')
        assert isinstance(op, scaluq.GroupedPauliOperator)
        assert op.n_groups() == 1
        assert op.is_cached() == bool(budget)
        out = x.copy()
        out.set_zero_norm_state()
        op.apply_to_state(x, out)
        assert out.get_amplitudes() == pytest.approx(expected.get_amplitudes())
        before = x.get_amplitudes()
        op.apply_to_state(x, x)
        assert x.get_amplitudes() == pytest.approx([a + b for a, b in zip(before, expected.get_amplitudes())])
        x.load(before)
