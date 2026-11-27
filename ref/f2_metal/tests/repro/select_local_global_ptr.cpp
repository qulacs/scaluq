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
