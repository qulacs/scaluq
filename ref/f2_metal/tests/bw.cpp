#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <cstdlib>
int main(int argc,char**argv){ setvbuf(stdout,nullptr,_IONBF,0);
  int coarse = argc>1 && atoi(argv[1]);
  sycl::property_list pl = coarse ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::AdaptiveCpp_coarse_grained_events{}} : sycl::property_list{sycl::property::queue::in_order{}};
  sycl::queue q{sycl::gpu_selector_v, pl};
  for (int lg : {16, 20, 24, 26, 28}) {
    size_t n = 1ull<<lg; float* x=sycl::malloc_device<float>(n,q); float* y=sycl::malloc_device<float>(n,q);
    q.fill(x,1.0f,n); q.fill(y,2.0f,n); q.wait();
    for(int w=0;w<5;++w) q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){ y[i]=0.5f*x[i]+y[i];}); q.wait();
    int R = lg<=20 ? 200 : 30;
    auto t0=std::chrono::steady_clock::now(); for(int r=0;r<R;++r) q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){ y[i]=0.5f*x[i]+y[i];}); q.wait();
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/R;
    std::printf("coarse=%d n=2^%d: %.3f ms/launch, %.1f GB/s\n", coarse, lg, dt*1e3, 3.0*4*n/dt/1e9);
    sycl::free(x,q); sycl::free(y,q);
  }
}
