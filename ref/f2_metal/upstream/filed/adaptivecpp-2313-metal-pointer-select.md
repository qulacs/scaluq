## Summary

On the Metal backend, a kernel that chooses at run time between a local (threadgroup) pointer and a global (device) pointer fails in the Metal emitter:

```
jit::compile: Encountered errors:
0: LLVMToMetal: MetalEmitter failed: Error: Conflicting address space information in PHI node
metal_queue: Code object construction failed
```

MSL has no generic address space, so this is probably a fundamental limitation and not a bug. We hit it through Kokkos: `team.team_scratch(level)` with a run-time `level` produces exactly this pattern. We worked around it by making the level a compile-time constant.

**Environment:** AdaptiveCpp `develop` at 6c923f68 (the current tip), generic SSCP compiler, Metal backend (`-DWITH_METAL_BACKEND=ON`), Homebrew LLVM 21.1.8, macOS 26.7.1, Apple M3 Ultra.

## Reproducer

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
// Choose at run time between a local (threadgroup) and a global (device) buffer.
int main(int argc, char** argv) {
  const int use_global = argc > 1 ? std::atoi(argv[1]) : 0;
  sycl::queue q{sycl::gpu_selector_v};
  const size_t wg = 64;
  float* g = sycl::malloc_device<float>(wg, q);
  float* out = sycl::malloc_shared<float>(wg, q);
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> loc(wg, h);
    h.parallel_for(sycl::nd_range<1>(wg, wg), [=](sycl::nd_item<1> it) {
      size_t l = it.get_local_id(0);
      float* buf = use_global ? g : &loc[0];
      buf[l] = float(l);
      sycl::group_barrier(it.get_group());
      out[l] = buf[wg - 1 - l];
    });
  }).wait();
  std::printf("out[0] = %g (expect 63)\n", out[0]);
}
```

Output: the JIT error above, then `out[0] = 0 (expect 63)`, and then the asynchronous error handler terminates the program.

## Request

- Mention this limitation in the Metal backend documentation, because the same source works on the other backends.
- If possible, make the error name the source location of the pointer select or PHI. In a large code base such as Kokkos, finding the offending code from the current message takes a long time.
