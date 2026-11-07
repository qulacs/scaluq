# Draft discussion for Kokkos (not filed yet)

**Title:** Building the SYCL backend with AdaptiveCpp (generic SSCP), e.g. for Apple GPUs via Metal

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

- **Two-pass reduce and scan.**
  - The single-pass "last work-group" scheme, an `atomic_ref<acq_rel, device>` counter followed by reading the other groups' partials, returns wrong results on the Metal backend.
  - AdaptiveCpp lowers RMW atomics as relaxed on Metal, and plain stores from other threadgroups were not visible even after a device-scope fence.
  - Under `__ACPP__` the patch instead writes partials and launches a second single-work-group kernel on the same in-order queue. This covers Range, MDRange, Team and Range scan.
  - Would a two-pass mode be acceptable upstream, selected by a macro or by a backend capability?
- **Destruction order in `~SYCLInternal`.**
  - `m_team_scratch_event[]` is declared before `m_queue`, so the queue is destroyed first.
  - AdaptiveCpp crashes when such an event is destroyed later. We will report this to AdaptiveCpp.
  - Releasing the events explicitly in the destructor (one line) is harmless for DPC++ and could go upstream independently.

## Status

- Tested only on Apple M3 Ultra, macOS 26.7.1, with AdaptiveCpp develop `6c923f68`.
- Performance is not measured yet.
- f64 is not possible on Apple GPUs.
