import pytest
from scaluq import default


def test_pauli_apply_add_and_operator_apply():
    m = default.f64
    x = m.StateVector(1)
    x.load([1, 2])
    p = m.PauliOperator('Y 0', 2 + 1j)
    expected = x.copy()
    applied = x.copy()
    p.apply_to_state(applied)
    expected.add_state_vector_with_coef(0.5 - 0.25j, applied)
    out = x.copy()
    p.apply_add(x, out, 0.5 - 0.25j)
    assert out.get_amplitudes() == pytest.approx(expected.get_amplitudes())
    op = m.Operator([p])
    op.apply_to_state(out)
    p.apply_to_state(expected)
    assert out.get_amplitudes() == pytest.approx(expected.get_amplitudes())


def test_pauli_apply_add_checks_and_y_phase():
    m = default.f64
    x = m.StateVector(1)
    x.load([0, 1])
    out = x.copy()
    out.set_zero_norm_state()
    m.PauliOperator('Y 0').apply_add(x, out)
    assert out.get_amplitudes() == pytest.approx([-1j, 0])
    with pytest.raises(RuntimeError):
        m.PauliOperator('X 1').apply_add(x, out)
    with pytest.raises(RuntimeError):
        m.PauliOperator('X 0').apply_add(x, m.StateVector(2))
    scalar = m.StateVector(0)
    scalar.load([3])
    m.Operator([]).apply_to_state(scalar)
    assert scalar.get_amplitudes() == pytest.approx([0])
