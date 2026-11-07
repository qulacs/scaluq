#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
// mode 0: nd_range+local mem, waited; 1: not waited; 2: event copied into heap struct destroyed after queue
struct Holder { sycl::event ev[4]; sycl::queue q; };  // events declared before queue, like SYCLInternal
int main(int argc, char** argv) {
  int mode = argc > 1 ? std::atoi(argv[1]) : 0;
  sycl::event ev;
  if (mode == 2) {
    auto* h = new Holder{{}, sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order{}}};
    float* p = sycl::malloc_device<float>(256, h->q);
    for (int i = 0; i < 4; ++i)
      h->ev[i] = h->q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> loc(64, cgh);
        cgh.parallel_for(sycl::nd_range<1>(256, 64), [=](sycl::nd_item<1> it) {
          loc[it.get_local_id(0)] = it.get_global_id(0);
          sycl::group_barrier(it.get_group());
          p[it.get_global_id(0)] = loc[63 - it.get_local_id(0)];
        });
      });
    h->q.wait();
    sycl::free(p, h->q);
    std::printf("deleting holder (queue member destroyed before events)\n"); std::fflush(stdout);
    delete h;
    std::printf("ok\n"); return 0;
  }
  {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    float* p = sycl::malloc_device<float>(256, q);
    ev = q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> loc(64, cgh);
      cgh.parallel_for(sycl::nd_range<1>(256, 64), [=](sycl::nd_item<1> it) {
        loc[it.get_local_id(0)] = it.get_global_id(0);
        sycl::group_barrier(it.get_group());
        p[it.get_global_id(0)] = loc[63 - it.get_local_id(0)];
      });
    });
    if (mode == 0) q.wait();
  }
  std::printf("queue destroyed, destroying event\n"); std::fflush(stdout);
  ev = sycl::event{};
  std::printf("ok\n");
}
