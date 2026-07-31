# Scaluq/Qulacs 4x4 SIMD path benchmark

This benchmark separates Scaluq's 4x4 dense SIMD kernels into the following
uncontrolled target placements:

- `low`: `{0,1}` (Scaluq F32 and Qulacs F64)
- `middle`: `{0,2}` (Scaluq F32/F64 and Qulacs F64)
- `high`: `{2,3}` (Scaluq F32/F64 and Qulacs F64)

Scaluq F64 has two complex lanes and therefore has no 4x4 `low` path. The low
plot deliberately omits Scaluq F64 and labels the remaining precision
difference explicitly.

Configure and build:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSCALUQ_CPU_NATIVE=ON \
  -DSCALUQ_USE_EXE=ON \
  -DSCALUQ_BUILD_QULACS_SIMD_PATH_BENCHMARK=ON
cmake --build build --target simd_path_qulacs_benchmark -j4
```

Run and plot:

```sh
mkdir -p benchmark-results
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
QULACS_PARALLEL_NQUBIT_THRESHOLD=13 \
  ./build/exe/simd_path_qulacs_benchmark \
  --min-qubits 4 --max-qubits 24 \
  --warmup 5 --iterations 20 \
  --output benchmark-results/simd-path-comparison.csv

python3 exe/plot_simd_path_comparison.py \
  benchmark-results/simd-path-comparison.csv \
  --output-dir benchmark-results
```

The generated images are:

- `benchmark-results/comparison-low.png`
- `benchmark-results/comparison-middle.png`
- `benchmark-results/comparison-high.png`

For each precision and SIMD path, the benchmark runs four complete
`min-qubits` through `max-qubits` sections:

1. Scaluq in ascending qubit order
2. Qulacs in ascending qubit order
3. Qulacs in descending qubit order
4. Scaluq in descending qubit order

The reported value for each implementation is the median of its ascending and
descending section results. Qulacs state storage is first-touched in parallel
with the same OpenMP placement used by its kernel, avoiding a serial
first-touch NUMA bias at large state sizes.

Scaluq uses `HostSerial` below 13 qubits and `Default` from 13 qubits onward,
matching Qulacs' internal OpenMP threshold. Set
`QULACS_PARALLEL_NQUBIT_THRESHOLD` to change this threshold for both
implementations. Speedup is `Qulacs median / Scaluq median`, so values above
one mean Scaluq is faster. The F64 rows compare equal precision; F32 rows
compare Scaluq F32 against Qulacs F64 and should be treated as a
mixed-precision reference.
