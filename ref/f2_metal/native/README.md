# native_bench: native Metal baseline for the scaluq SYCL/Kokkos port

Single precision state vector, interleaved `float2` (complex<float>). MSL is compiled at
runtime (`newLibraryWithSource:`), so no offline `metal` toolchain is needed.

Build: `./build.sh` (or `make`). Run: `./native_bench [n ...]` (default `16 20 24 26 28`).

## Storage mode
State and triad arrays are `MTLResourceStorageModePrivate` (GPU-optimal, no CPU coherence
overhead). Initial data and verification results go through a `Shared` staging buffer and
blit copies. Only the 4-byte norm result is read directly from a Shared buffer.

## Kernels
- `launch_empty`: empty kernel, 256 threads. `single_cb_*`: one command buffer per dispatch
  (commit + waitUntilCompleted), R=1000. `batched_*`: 1000 dispatches in one encoder, one wait.
- `triad`: `y = 0.5*x + y`, float arrays of 2^n; 12 B/element.
- `H`: one thread per pair (insert zero bit at t); t = 0 and n-1; 32 B per pair.
- `CNOT`: one thread per index with control=1,target=0 (two zero bits inserted, control set),
  swaps with target=1 partner; (c,t) = (n-1,0) and (0,n-1); 32 B per active pair, N/4 pairs.
- `norm`: sum |a_i|^2. Pass 1: 2048 threadgroups x 256 threads, grid-stride loop, `simd_sum` +
  threadgroup memory, writes partials; pass 2: one threadgroup reduces partials. GB/s counts 8 B/elem.
  Result checked against a CPU double sum (relative error in the note column).

Batched variants encode R dispatches (R = 2^30>>n clamped to [10,200]) into one command
buffer and encoder; dispatches are serialized by Metal hazard tracking on the shared buffer.
Kernels are warmed up (3 dispatches) before timing.

## Output
TSV on stdout: `kernel  n_qubits  variant  ms_per_op  GBps  note`. Variants ending `_wall` are
steady_clock around commit/wait divided by ops; `_gpu` use GPUEndTime-GPUStartTime. Lines
starting with `#` are comments (device name, correctness PASS/FAIL). Correctness is run once at
n = min(20, largest n): H (t=0, mid, n-1) and CNOT (3 pairs) compared element-wise and by norm
with a CPU reference. Exit code 2 if correctness fails.
