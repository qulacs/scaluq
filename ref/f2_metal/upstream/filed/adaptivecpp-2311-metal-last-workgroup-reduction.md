## Summary

On the Metal backend, the common "last work-group finishes the reduction" pattern returns wrong results once there are many work-groups. In this pattern, each group writes a partial result and then increments an `acq_rel`, device-scope counter, and the group that sees the final count reads all the partials. Kokkos and oneDPL use this pattern for reductions.

There seem to be two parts:

1. **RMW atomics ignore `memory_order` and `memory_scope`.** In `src/libkernel/sscp/metal/atomic.cpp`, load and store wrap a relaxed atomic with `atomic_thread_fence` according to the requested order. The read-modify-write builtins (`fetch_add/sub/and/or/xor/min/max`, `exchange`, `compare_exchange_*`) drop the order and scope arguments. They always emit `memory_order_relaxed` with no fence; for example, `__acpp_sscp_atomic_fetch_add_i32` emits `atomic_fetch_add_explicit(..., memory_order_relaxed)`. So `atomic_ref<unsigned, memory_order::acq_rel, memory_scope::device>` behaves as relaxed.
2. **Plain stores from another work-group stay invisible, even with an explicit fence.** With `sycl::atomic_fence(memory_order::seq_cst, memory_scope::device)` after the store and before the loads, the result is still wrong. Only atomic loads and stores of the partials give correct results.

**Environment:** AdaptiveCpp `develop` at 6c923f68 (the current tip), generic SSCP compiler, Metal backend (`-DWITH_METAL_BACKEND=ON`), Homebrew LLVM 21.1.8, macOS 26.7.1, Apple M3 Ultra.

## Reproducer

Each work-group computes its size (256), and the last group sums the partials. The argument selects plain stores and loads (`0`), plain stores and loads plus `seq_cst` device fences (`1`), or relaxed atomic stores and loads (`2`).

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){ setvbuf(stdout,nullptr,_IONBF,0); int fence = argc>1 ? atoi(argv[1]) : 0;
  sycl::queue q{sycl::gpu_selector_v}; const size_t wg=256;
  for (size_t ng : {1, 2, 4, 16, 256}) {
    size_t n=ng*wg; int* partial=sycl::malloc_device<int>(ng,q); unsigned* flag=sycl::malloc_device<unsigned>(1,q); int* result=sycl::malloc_shared<int>(1,q);
    q.memset(flag,0,4).wait(); *result=-1;
    q.submit([&](sycl::handler& h){ sycl::local_accessor<unsigned,1> done(1,h); sycl::local_accessor<int,1> loc(wg,h);
      h.parallel_for(sycl::nd_range<1>(n,wg),[=](sycl::nd_item<1> it){ size_t l=it.get_local_id(0), g=it.get_group(0);
        int v = sycl::reduce_over_group(it.get_group(), 1, sycl::plus<int>());
        if(l==0){ if(fence==2) sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(partial[g]).store(v); else partial[g]=v;
          if(fence==1) sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
          sycl::atomic_ref<unsigned,sycl::memory_order::acq_rel,sycl::memory_scope::device,sycl::access::address_space::global_space> a(*flag);
          done[0] = ++a; }
        sycl::group_barrier(it.get_group());
        if(done[0]==ng){ if(fence==1) sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
          int s = 0; for(size_t k=l;k<ng;k+=wg) s+= (fence==2) ? sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>(partial[k]).load() : partial[k];
          s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<int>()); if(l==0) *result=s; }
      });}).wait();
    std::printf("fence=%d groups=%zu: result=%d expect %zu %s\n", fence, ng, *result, n, (*result==(int)n)?"OK":"WRONG");
  }
}
```

Results (3 runs each; only the 256-group case is shown, smaller counts were always correct):

| mode | groups = 256, expected 65536 |
|---|---|
| 0: plain store/load | 35840, 16384, 36864 |
| 1: plain + `seq_cst` device fence | 47104, 36864, 16384 |
| 2: relaxed atomic store/load | 65536, 65536, 65536 |

## Expected

- RMW builtins should emit fences for acquire, release, acq_rel and seq_cst, as load and store already do. Alternatively, the backend could document that only relaxed RMW is supported and report it through `info::device::atomic_memory_order_capabilities`, which is not implemented yet (#2310).
- It would help to document which cross-work-group visibility guarantees the Metal backend gives. Case 1 suggests that fixing the RMW lowering alone may not be enough.

## Context

We hit this while running Kokkos (SYCL backend built with acpp) on Apple GPUs for the quantum simulator scaluq. As a workaround, our local Kokkos patch does reductions in two kernels: one writes the partials, and a second single-group kernel combines them. That is correct on Metal.
