import statistics
import time

import scaluq
from scaluq.default.f64 import Circuit, StateVector
from scaluq.default.f64.gate import H, RX, RZ, X


def build_circuit(n_qubits: int, depth: int) -> Circuit:
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


def bench_update(circuit: Circuit, state: StateVector, n_iters: int) -> list[float]:
    times = []
    for _ in range(n_iters):
        state.set_zero_state()
        scaluq.synchronize()
        t0 = time.perf_counter()
        circuit.update_quantum_state(state, {})
        scaluq.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def print_stats(label: str, times: list[float]) -> None:
    mean_s = statistics.mean(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"{label}: {mean_s * 1e3:.3f} ms (stdev {stdev_s * 1e3:.3f} ms)")


def main() -> int:
    n_qubits = 20
    depth = 50
    n_warmup = 2
    n_iters = 8

    try:
        import scaluq.host  # noqa: F401
        has_cuda = True
    except Exception:
        has_cuda = False

    print("ScaluQ CUDA build:", "yes" if has_cuda else "no")
    print(f"Circuit: n_qubits={n_qubits}, depth={depth}")

    circuit = build_circuit(n_qubits, depth)
    state = StateVector(n_qubits)

    for _ in range(n_warmup):
        state.set_zero_state()
        circuit.update_quantum_state(state, {})
        scaluq.synchronize()

    times_default = bench_update(circuit, state, n_iters)
    print_stats("Default stream (single task)", times_default)

    stream_count = 4
    streams = scaluq.create_streams([1.0] * stream_count)
    print("ConcurrentStream enabled:", "yes" if streams else "no")

    # Independent tasks on multiple streams
    states = [StateVector(n_qubits) for _ in range(stream_count)]
    for idx, state in enumerate(states):
        state.set_execution_space(streams[idx])

    # Warmup: sequential (no overlap)
    for _ in range(n_warmup):
        for state in states:
            state.set_zero_state()
        for state in states:
            circuit.update_quantum_state(state, {})
            scaluq.synchronize(streams)

    # Measure sequential execution (all tasks, with fences)
    times_seq = []
    for _ in range(n_iters):
        for state in states:
            state.set_zero_state()
        scaluq.synchronize(streams)
        t0 = time.perf_counter()
        for idx, state in enumerate(states):
            circuit.update_quantum_state(state, {})
            scaluq.synchronize(streams[idx])
        t1 = time.perf_counter()
        times_seq.append(t1 - t0)
    print_stats(f"ConcurrentStream ({stream_count} tasks, sequential)", times_seq)

    # Warmup: concurrent (enqueue all, fence once)
    for _ in range(n_warmup):
        for state in states:
            state.set_zero_state()
        for state in states:
            circuit.update_quantum_state(state, {})
        scaluq.synchronize(streams)

    # Measure concurrent execution (enqueue all, fence once)
    times_conc = []
    for _ in range(n_iters):
        for state in states:
            state.set_zero_state()
        scaluq.synchronize(streams)
        t0 = time.perf_counter()
        for state in states:
            circuit.update_quantum_state(state, {})
        scaluq.synchronize(streams)
        t1 = time.perf_counter()
        times_conc.append(t1 - t0)
    print_stats(f"ConcurrentStream ({stream_count} tasks, overlapped)", times_conc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
