## Context

We maintain scaluq, a quantum circuit simulator built on Kokkos. We tried running it on Apple
Silicon GPUs by building Kokkos 5.2.0's SYCL backend with AdaptiveCpp (`acpp`, generic SSCP
compiler, Metal backend).

With about 300 lines of changes to Kokkos, all 229 single-precision scaluq tests pass on an
Apple M3 Ultra. The full patch is in `ref/f2_metal/patches/kokkos-5.2.0-acpp-metal.patch` of
qulacs/scaluq#463.

We would like to know whether upstream is interested in AdaptiveCpp support. If so, which
parts would you accept, and in what form?

## What had to change

### Build and macros

- The DPC++ flags (`-fsycl`, `-fsycl-targets`, ...) must be skipped (`kokkos_arch.cmake`).
- `__SYCL_DEVICE_ONLY__` is not defined by AdaptiveCpp's single-pass SSCP compiler.
  - `KOKKOS_IF_ON_DEVICE` / `DESUL_IF_ON_DEVICE` need `if (__acpp_sscp_is_device)` instead.
  - Without this, host-only code paths were taken in kernels ("inaccessible memory space").
- `Kokkos::printf` is made a no-op. `remquo` is excluded. `nan` uses `__builtin_nan`.
- `Kokkos_BitManipulation.hpp` needed the generic `Op<true,true>` path.

### DPC++/oneAPI-only APIs

- `kernel_bundle` / `get_kernel_id` / `kernel_device_specific` queries are not available.
  - In ParallelReduce, ParallelScan and the Team policies, fixed values are used instead (work-group size multiple 32, max 256, sub-group 32).
- `sub_group_mask` ballot in desul's SYCL atomics is replaced by a `while (!done)` loop.
- `group_local_memory_for_overwrite` is replaced by reusing `m_team_reduce`.
- `this_nd_item` (used by `UniqueToken` and `Random_UniqueIndex`) is replaced by AdaptiveCpp's SSCP work-item builtins.

### Algorithms

- **`Random_XorShift64_Pool::get_state()` on GPUs without forward-progress guarantees.**
  - On Metal, a kernel that calls `get_state()` from far more work-items than the pool has states (for example 2^24 work-items) did not finish in 10 minutes.
  - After the process was killed, the kernel kept the GPU at 100% until the OS was rebooted.
  - The suspected cause is that `get_state()` spins on a per-state lock with `atomic_compare_exchange`. Metal, like pre-Volta CUDA, does not guarantee that the lock holder makes progress while other lanes of the same SIMD group spin.
  - Smaller launches (up to about 2^20 work-items) work. Should the SYCL backend avoid spinning here, for example by deriving the state directly from the work-item id?


- **Two-pass reduce and scan.**
  - The single-pass "last work-group" scheme, an `atomic_ref<acq_rel, device>` counter followed by reading the other groups' partials, returns wrong results on the Metal backend.
  - AdaptiveCpp lowers RMW atomics as relaxed on Metal, and plain stores from other threadgroups were not visible even after a device-scope fence.
  - Under `__ACPP__` the patch instead writes partials and launches a second single-work-group kernel on the same in-order queue. This covers Range, MDRange, Team and Range scan.
  - Would a two-pass mode be acceptable upstream, selected by a macro or by a backend capability?
- **Destruction order in `~SYCLInternal`.**
  - `m_team_scratch_event[]` is declared before `m_queue`, so the queue is destroyed first.
  - With AdaptiveCpp, every program that ran a `TeamPolicy` with scratch memory crashed in `Kokkos::finalize()`, in `hipsycl::rt::dag_node::~dag_node()` called from `~SYCLInternal`.
  - Releasing the events explicitly at the start of the destructor (one line) fixes it. This should be harmless for DPC++ and could go upstream on its own.
  - A standalone SYCL program that destroys an event after its in-order queue does not crash. So the exact trigger inside Kokkos is still unclear.

## Status

- Tested only on an Apple M3 Ultra with macOS 26.7.1 and AdaptiveCpp develop `6c923f68`. f64 is not possible on Apple GPUs.
- **Performance** (idle GPU, f32):
  - For large states (2^24 to 2^28 amplitudes), scaluq's gates through Kokkos reach about 700 GB/s, the same as hand-written native Metal kernels.
  - Kernel launch latency is the main gap: about 65 µs per small kernel through Kokkos, compared with about 1 µs for batched native Metal dispatches. Plain SYCL drops from 152 µs to 9 µs per launch with AdaptiveCpp's `AdaptiveCpp_coarse_grained_events` queue property. We plan to try that property on the queue Kokkos creates.
  - The two-pass reductions take about 1.4× the time of a native two-pass reduction on large inputs.
- Related AdaptiveCpp issues:
  - AdaptiveCpp/AdaptiveCpp#2311: the last-work-group reduction pattern is wrong on Metal. This is the reason for the two-pass reduce and scan.
  - AdaptiveCpp/AdaptiveCpp#2310: the atomic capability queries are missing, so the backend cannot be detected at run time.
  - AdaptiveCpp/AdaptiveCpp#2312: `__acpp_sscp_get_num_subgroups` is missing on Metal.
  - AdaptiveCpp/AdaptiveCpp#2313: a run-time select between local and global pointers fails on Metal. Kokkos' `team_scratch(level)` with a run-time level triggers it.
