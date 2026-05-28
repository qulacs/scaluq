# About Scaluq

Scaluq is a newly redeveloped Python/C++ library based on the quantum circuit simulator [Qulacs](https://github.com/qulacs/qulacs).
It enables high-speed simulation of large-scale quantum circuits, noisy quantum circuits, and parametric quantum circuits.
Scaluq is licensed under the [MIT license](https://github.com/qulacs/qulacs/blob/master/LICENSE).

# Feature

Compared to [Qulacs](https://github.com/qulacs/qulacs), the following improvements have been made:

- Implementation based on [Kokkos](https://github.com/kokkos/kokkos) allows seamless switching between execution environments (CPU/GPU) without requiring code changes.
- Improved execution speed.
- Pointers are hidden from users, making the code simpler and safer to write.
- Integration of [nanobind](https://github.com/wjakob/nanobind) enables more compact and faster Python bindings.
- Provides a faster interface for the case where the same circuit is applied to multiple quantum states.

# Performance

The time for simulating quantum circuits (Method: Benchmark execution time to apply CX, RX and RZ gates (averaged over every target qubit)) is compared with several quantum circuit simulators in 2026.

See [the benchmark repository](https://github.com/Qulacs-Osaka/benchmark-scaluq).

## Single State Vector Update (January 2026)

### CPU result
![Single State Vector Update (CPU)](https://github.com/Qulacs-Osaka/benchmark-scaluq/blob/main/benchmark/multiple-gate/multithread/image/circuit.png)

### GPU result
![Single State Vector Update (GPU)](https://github.com/Qulacs-Osaka/benchmark-scaluq/blob/main/benchmark/multiple-gate/gpu/image/circuit.png)

## Batched State Vector Update (May 2026)

### Varying batch size (#qubits=16)
![Batched State Vector Update (batch sweep)](https://github.com/Qulacs-Osaka/benchmark-scaluq/blob/main/benchmark/batch/image/batch_sweep.png)

### Varying #qubits (batch size=100)
![Batched State Vector Update (qubits sweep)](https://github.com/Qulacs-Osaka/benchmark-scaluq/blob/main/benchmark/batch/image/qubits_sweep.png)

## Build Requirements

- Ninja ≥ 1.10
- GCC ≥ 11 (≥ 13 if not using CUDA)
- CMake ≥ 3.24
- CUDA ≥ 12.6 (only when using CUDA)
- Python ≥ 3.10 (only when using Python)

When using CUDA, use a host compiler version supported by your CUDA toolkit (see the CUDA Installation Guide Host Compiler Support Policy).

Note: It may work with lower versions, but this has not been verified.

## Runtime Requirements

- CUDA ≥ 12.6 (only when using CUDA)

Note: It may work with lower versions, but this has not been verified.
