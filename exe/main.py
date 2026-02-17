import math
import statistics
import time

import scaluq
from scaluq.default.f64 import Operator
from scaluq.default.f64 import StateVector
from scaluq.default.f64.gate import H, RX, RZ, X
from scaluq.default.f64 import Circuit, PauliOperator


def synchronize_all(streams):
    for stream in streams:
        scaluq.synchronize(stream)


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


def mean_ms(samples):
    return statistics.mean(samples) * 1e3


def build_circuit(n_qubits, depth):
    circuit = Circuit(n_qubits)
    for layer in range(depth):
        for q in range(n_qubits):
            if layer % 3 == 0:
                circuit.add_gate(H(q))
            elif layer % 3 == 1:
                circuit.add_gate(RX(q, 0.123 + 0.01 * layer))
            else:
                circuit.add_gate(RZ(q, 0.456 + 0.01 * layer))
        for q in range(0, n_qubits - 1, 2):
            circuit.add_gate(X(q + 1, controls=[q]))
    return circuit


def bench_stream_overlap(circuit, streams, n_qubits, n_warmup=2, n_iters=6):
    states = [StateVector(streams[i], n_qubits) for i in range(len(streams))]

    # Warmup
    for _ in range(n_warmup):
        for st in states:
            st.set_zero_state()
            circuit.update_quantum_state(st, {})
            scaluq.synchronize(st.concurrent_stream())

    seq = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        for st in states:
            st.set_zero_state()
            circuit.update_quantum_state(st, {})
            scaluq.synchronize(st.concurrent_stream())
        t1 = time.perf_counter()
        seq.append(t1 - t0)

    ovl = []
    for _ in range(n_iters):
        for st in states:
            st.set_zero_state()
        synchronize_all(streams)
        t0 = time.perf_counter()
        for st in states:
            circuit.update_quantum_state(st, {})
        synchronize_all(streams)
        t1 = time.perf_counter()
        ovl.append(t1 - t0)
    return seq, ovl


def main():
    has_cuda = False
    try:
        import scaluq.host  # noqa: F401
        has_cuda = True
    except Exception:
        pass

    print("CUDA build:", has_cuda)

    print("\n[1] Create streams")
    s0, s1 = create_default_streams([1.0, 1.0])
    print("default streams created:", bool(s0 and s1))

    print("\n[2] Copy across streams with explicit source sync")
    psi = StateVector(s0, 3)
    psi.set_Haar_random_state(7)
    scaluq.synchronize(s0)
    psi_s1 = psi.copy(s1)
    scaluq.synchronize(s1)
    amp0 = psi.get_amplitudes()
    amp1 = psi_s1.get_amplitudes()
    max_abs_err = max(abs(a - b) for a, b in zip(amp0, amp1))
    print("max |psi - copy|:", max_abs_err)

    print("\n[3] set_concurrent_stream behavior")
    psi.set_concurrent_stream(s1)
    psi.normalize()
    scaluq.synchronize(s1)
    norm = psi.get_squared_norm()
    print("squared norm after normalize on s1:", norm)
    if not math.isclose(norm, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("normalize() on switched stream failed")

    print("\n[4] Operator uses StateVector stream for state operations")
    try:
        op = Operator(s0, [PauliOperator("X 0")])
    except TypeError:
        # Compatibility for old Python package that does not expose stream ctor.
        op = Operator([PauliOperator("X 0")])
        if hasattr(op, "set_concurrent_stream"):
            op.set_concurrent_stream(s0)
    st = StateVector(s1, 1)
    st.set_zero_state()
    op.apply_to_state(st)
    scaluq.synchronize(s1)
    amps = st.get_amplitudes()
    print("state after X on |0>:", amps)

    print("\n[5] Stream-overlap benchmark (not guaranteed to speed up)")
    circuit = build_circuit(n_qubits=18, depth=40)
    seq, ovl = bench_stream_overlap(circuit, [s0, s1], n_qubits=18, n_warmup=1, n_iters=4)
    print(f"sequential avg: {mean_ms(seq):.3f} ms")
    print(f"overlapped avg: {mean_ms(ovl):.3f} ms")

    if has_cuda:
        print("\n[6] Type mismatch check (default object + host stream)")
        if hasattr(scaluq, "create_host_streams"):
            h0 = create_host_streams([1.0])[0]
            caught = False
            try:
                _ = psi.copy(h0)
            except RuntimeError as e:
                caught = True
                print("caught RuntimeError:", str(e))
            print("type mismatch raised:", caught)
        else:
            print("skipped: create_host_streams() is not available in this build")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
