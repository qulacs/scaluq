## Summary

The SYCL 2020 device queries `info::device::atomic_memory_order_capabilities`, `atomic_memory_scope_capabilities`, `atomic_fence_order_capabilities` and `atomic_fence_scope_capabilities` are not available:

```
error: no member named 'atomic_memory_order_capabilities' in namespace 'hipsycl::sycl::info::device'
```

Portable libraries such as Kokkos and oneDPL could use these queries to choose another algorithm when a backend supports only relaxed RMW atomics. The Metal backend currently lowers all RMW atomics as relaxed; see #2311.

**Environment:** AdaptiveCpp `develop` at 6c923f68 (the current tip), generic SSCP compiler, Metal backend (`-DWITH_METAL_BACKEND=ON`), Homebrew LLVM 21.1.8, macOS 26.7.1, Apple M3 Ultra.

## Reproducer

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto orders = q.get_device().get_info<sycl::info::device::atomic_memory_order_capabilities>();
  std::printf("%zu orders\n", orders.size());
}
```
