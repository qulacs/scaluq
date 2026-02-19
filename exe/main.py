import scaluq
from scaluq.default.f64 import OperatorBatched, PauliOperator, StateVectorBatched


def create_default_streams(weights):
    if hasattr(scaluq, "create_default_streams"):
        return scaluq.create_default_streams(weights)
    if hasattr(scaluq, "create_streams"):
        return scaluq.create_streams(weights)
    raise RuntimeError("No stream creation API found in scaluq module")


def create_host_streams(weights):
    if hasattr(scaluq, "create_host_streams"):
        return scaluq.create_host_streams(weights)
    raise RuntimeError("create_host_streams is not available in this scaluq build")


def assert_complex_close(name, value, expected, tol=1e-10):
    if abs(value - expected) > tol:
        raise RuntimeError(
            f"{name}: expected {expected}, got {value}, abs_err={abs(value - expected)}"
        )


def test_state_vector_batched_utilities(s0, s1):
    print("\n[1] StateVectorBatched utilities")
    states = StateVectorBatched(s0, 2, 2)
    states.set_zero_norm_state()

    changed = complex(0.25, -0.5)
    view_state = states.view_state_vector_at(s1, 1)
    view_state.set_amplitude_at(3, changed)
    scaluq.synchronize(s1)

    got = states.get_state_vector_at(s0, 1).get_amplitude_at(3)
    assert_complex_close("view reflected to batched", got, changed)
    got_by_get = states.get_state_vector_at(s0, 1).get_amplitude_at(3)
    assert_complex_close("get_state_vector_at(stream, id)", got_by_get, changed)

    other_batch = states.get_state_vector_at(s0, 0).get_amplitude_at(3)
    assert_complex_close("other batch remains unchanged", other_batch, 0j)

    copied = states.get_state_vector_at(s1, 1)
    copied.set_amplitude_at(3, complex(0.9, 0.1))
    scaluq.synchronize(s1)

    got_after_copy_edit = states.get_state_vector_at(s0, 1).get_amplitude_at(3)
    assert_complex_close("copy does not modify original", got_after_copy_edit, changed)
    print("StateVectorBatched view/copy checks passed.")


def test_operator_batched_utilities(s0, s1):
    print("\n[2] OperatorBatched utilities")
    operators = OperatorBatched(
        s0,
        [
            [PauliOperator("X 0", 1.0)],
            [PauliOperator("Z 0", 2.0)],
        ],
    )

    view_op = operators.view_operator_at(s1, 0)
    view_op *= 3.0
    scaluq.synchronize(s1)
    coef_after_view = operators.get_operator_at(s0, 0).get_terms()[0].coef()
    assert_complex_close("view reflected to batched operator", coef_after_view, 3 + 0j)

    copied_op = operators.get_operator_at(s1, 1)
    copied_op *= 5.0
    scaluq.synchronize(s1)
    coef_original = operators.get_operator_at(s0, 1).get_terms()[0].coef()
    assert_complex_close("copied operator does not modify original", coef_original, 2 + 0j)

    caught = False
    try:
        view_op.optimize()
    except RuntimeError:
        caught = True
    if not caught:
        raise RuntimeError("view operator must reject optimize()")
    print("OperatorBatched view/copy checks passed.")


def main():
    print("[0] Create streams")
    s0, s1 = create_default_streams([1.0, 1.0])
    print("default streams created:", bool(s0 and s1))

    test_state_vector_batched_utilities(s0, s1)
    test_operator_batched_utilities(s0, s1)

    if hasattr(scaluq, "create_host_streams"):
        print("\n[3] Optional host stream API check")
        host_stream = create_host_streams([1.0])[0]
        print("host stream created:", host_stream is not None)

    print("\nAll utility checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
