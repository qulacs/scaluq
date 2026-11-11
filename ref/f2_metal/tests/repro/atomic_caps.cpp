#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto orders = q.get_device().get_info<sycl::info::device::atomic_memory_order_capabilities>();
  std::printf("%zu orders\n", orders.size());
}
