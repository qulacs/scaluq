# Draft issues for AdaptiveCpp (not filed yet)

Environment for all items: AdaptiveCpp develop `6c923f68`, generic SSCP compiler, Metal backend
(`-DWITH_METAL_BACKEND=ON`), LLVM 21 (Homebrew), macOS 26.7.1, Apple M3 Ultra.
Reproducers are in `ref/f2_metal/tests/` of qulacs/scaluq#463.

---

## 1. Metal: RMW atomics ignore memory_order and memory_scope

**Summary.** In `src/libkernel/sscp/metal/atomic.cpp`, load and store wrap a relaxed atomic with
`atomic_thread_fence` according to the requested order. The read-modify-write builtins
(`fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`, `fetch_min`, `fetch_max`,
`exchange`, `compare_exchange_*`, 23 builtins in total) drop the order and scope arguments and
always emit `memory_order_relaxed` with no fence.

**Effect.** `sycl::atomic_ref<unsigned, memory_order::acq_rel, memory_scope::device>` behaves as
relaxed. The usual "last work-group finishes the reduction" pattern (used by Kokkos and oneDPL)
reads partial results that are not yet visible and returns wrong sums for more than one
work-group (`lastblock.cpp`).

**Expected.** Emit fences around RMW operations the same way as for load and store, or document
that only relaxed RMW is supported and report that through
`info::device::atomic_memory_order_capabilities` (see item 4).

**Note.** Even with an explicit `sycl::atomic_fence(memory_order::seq_cst, memory_scope::device)`,
which is lowered to `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
thread_scope_device)`, plain (non-atomic) stores from another threadgroup were not visible in
the same dispatch; only atomic loads and stores were. So fixing the RMW lowering may not be enough
for the last-work-group pattern on Metal. It would help to state in the documentation which
cross-threadgroup visibility guarantees the Metal backend gives.

---

## 2. Metal: `__acpp_sscp_get_num_subgroups` is not implemented

**Summary.** `sub_group::get_group_range()` and `nd_item::get_sub_group().get_group_range()` call
`__acpp_sscp_get_num_subgroups`, which has no Metal implementation. JIT fails with an unresolved
builtin.

**Fix used locally** (appended to `src/libkernel/sscp/metal/subgroup.cpp`):

```cpp
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_num_subgroups() {
  const uint sg = __acpp_sscp_get_subgroup_max_size();
  const uint wg = __acpp_sscp_get_local_size_x() * __acpp_sscp_get_local_size_y() *
                  __acpp_sscp_get_local_size_z();
  return (wg + sg - 1) / sg;
}
```

Happy to open a PR with this.

---

## 3. Destroying an event after its in-order queue crashes (seen with Metal)

**Summary.** If a `sycl::event` from an in-order queue is destroyed after the last `sycl::queue`
handle is destroyed, the program crashes with SIGSEGV in `hipsycl::rt::dag_node::~dag_node()`.

**How it shows up.** Kokkos' `SYCLInternal` declares `sycl::event m_team_scratch_event[]` before
`sycl::queue m_queue`, so the queue is destroyed first and the events afterwards. Every program
that ran a Kokkos `TeamPolicy` with scratch memory crashed in `Kokkos::finalize()`.

Backtrace:

```
hipsycl::rt::dag_node::~dag_node()
Kokkos::Impl::SYCLInternal::~SYCLInternal()
Kokkos::SYCL::impl_finalize()
Kokkos::finalize()
```

**Expected.** As far as I understand SYCL 2020, an event may outlive the queue that produced it.

**Not yet done.** A minimal reproducer without Kokkos is still to be written.

---

## 4. SYCL 2020 atomic capability queries are missing

`info::device::atomic_memory_order_capabilities`, `atomic_memory_scope_capabilities`,
`atomic_fence_order_capabilities` and `atomic_fence_scope_capabilities` do not compile. Portable
code (Kokkos, oneDPL) cannot detect that a backend supports only relaxed RMW (item 1).

---

## 5. Metal: select between threadgroup and device pointers cannot be emitted

**Summary.** IR that chooses at run time between a local (threadgroup) pointer and a global
(device) pointer fails in the Metal emitter, because MSL has no generic address space. Kokkos'
`team.team_scratch(level)` with a run-time `level` produces this. Error:

```
LLVMToMetal: MetalEmitter failed
```

This is probably a fundamental limitation. It would help to:

- list it in the Metal backend documentation, and
- make the error message name the pointer select and the source location.

---

## 6. (Minor) Missing pieces used by Kokkos

These are listed only for context; they may already be known. Kokkos needed patches for:

- the free-function `this_nd_item` (`sycl::ext::oneapi::this_work_item::get_nd_item`) and
- `kernel_bundle` / `get_kernel_id` with `kernel_device_specific` queries.
