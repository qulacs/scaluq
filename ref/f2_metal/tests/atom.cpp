#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
template<class T> using aref = sycl::atomic_ref<T,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>;
int main(int argc,char**argv){ setvbuf(stdout,nullptr,_IONBF,0); int m=atoi(argv[1]); int dev = argc>2; size_t n= argc>3? atoi(argv[3]) : (1<<20);
  sycl::queue q{sycl::gpu_selector_v};
  auto alloc=[&](size_t b){ return dev? sycl::malloc_device(b,q) : sycl::malloc_shared(b,q); };
  void* p=alloc(64); q.memset(p,0,64).wait();
  auto t0=std::chrono::steady_clock::now();
  if(m==0){ auto* c=(std::uint32_t*)p; q.parallel_for(sycl::range<1>(n),[=](sycl::id<1>){ aref<std::uint32_t>(*c).fetch_add(1u);}).wait(); }
  if(m==1){ auto* c=(float*)p; q.parallel_for(sycl::range<1>(n),[=](sycl::id<1>){ aref<float>(*c).fetch_add(1.0f);}).wait(); }
  if(m==2){ auto* c=(std::uint64_t*)p; q.parallel_for(sycl::range<1>(n),[=](sycl::id<1>){ aref<std::uint64_t>(*c).fetch_add(1ull);}).wait(); }
  if(m==3){ auto* c=(std::uint64_t*)p; q.parallel_for(sycl::range<1>(n),[=](sycl::id<1>){ aref<std::uint64_t> a(*c); std::uint64_t o=a.load(); while(!a.compare_exchange_weak(o,o+1)){} }).wait(); }
  if(m==4){ auto* c=(std::uint32_t*)p; q.parallel_for(sycl::range<1>(n),[=](sycl::id<1>){ aref<std::uint32_t> a(*c); std::uint32_t o=a.load(); while(!a.compare_exchange_weak(o,o+1)){} }).wait(); }
  // m==5: 64-bit fetch_add on distinct addresses (low contention)
  if(m==5){ auto* c=(std::uint64_t*)alloc(8*1024); q.memset(c,0,8*1024).wait(); q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){ aref<std::uint64_t>(c[i[0]%1024]).fetch_add(1ull);}).wait(); std::uint64_t h; q.memcpy(&h,c,8).wait(); std::printf("m5 c[0]=%llu expect %zu\n",(unsigned long long)h,n/1024); }
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
  std::uint64_t h=0; q.memcpy(&h,p,8).wait();
  std::printf("mode %d dev=%d n=%zu: raw=0x%llx (%.3f ms incl. JIT)\n", m, dev, n, (unsigned long long)h, ms);
}
