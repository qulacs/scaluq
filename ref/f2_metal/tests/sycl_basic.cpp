#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstdlib>
int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IONBF,0); int T = argc>1? atoi(argv[1]) : -1;
  sycl::queue q{sycl::gpu_selector_v};
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  const std::uint64_t n = 1ull<<24;
  float* x = sycl::malloc_device<float>(n,q); float* y = sycl::malloc_device<float>(n,q);
  q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){ x[i]=1.0f; y[i]=(float)(i[0]%7);}).wait();
  q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){ y[i]=2.0f*x[i]+y[i];}).wait();
  double ref=0; for(std::uint64_t i=0;i<n;++i) ref += 2.0 + (double)(i%7);
  const size_t wg=256; float* part = sycl::malloc_shared<float>(n/wg,q); double t=0;
  if(T<0||T==1){ // reduction
  float* s = sycl::malloc_shared<float>(1,q); *s=0;
  q.submit([&](sycl::handler& h){ h.parallel_for(sycl::range<1>(n), sycl::reduction(s, sycl::plus<float>()), [=](sycl::id<1> i, auto& r){ r += y[i]; });}).wait();
  std::printf("reduction(sycl::reduction): %.6e ref %.6e relerr %.2e\n", *s, ref, std::fabs(*s-ref)/ref);
  }
  if(T<0||T==2){ // nd_range + local memory + group barrier + group reduce
  q.submit([&](sycl::handler& h){ sycl::local_accessor<float,1> loc(wg,h);
    h.parallel_for(sycl::nd_range<1>(n,wg),[=](sycl::nd_item<1> it){ size_t l=it.get_local_id(0); loc[l]=y[it.get_global_id(0)];
      sycl::group_barrier(it.get_group()); for(size_t st=wg/2;st>0;st/=2){ if(l<st) loc[l]+=loc[l+st]; sycl::group_barrier(it.get_group()); }
      if(l==0) part[it.get_group_linear_id()]=loc[0]; });}).wait();
  t=0; for(size_t g=0;g<n/wg;++g) t+=part[g];
  std::printf("local-mem tree reduce: %.6e relerr %.2e\n", t, std::fabs(t-ref)/ref);
  }
  if(T<0||T==3){ // group algorithm
  q.submit([&](sycl::handler& h){ h.parallel_for(sycl::nd_range<1>(n,wg),[=](sycl::nd_item<1> it){ float v=sycl::reduce_over_group(it.get_group(), y[it.get_global_id(0)], sycl::plus<float>()); if(it.get_local_id(0)==0) part[it.get_group_linear_id()]=v;});}).wait();
  t=0; for(size_t g=0;g<n/wg;++g) t+=part[g];
  std::printf("reduce_over_group: %.6e relerr %.2e\n", t, std::fabs(t-ref)/ref);
  }
  if(T<0||T==4){ // atomics 32 / 64-bit
  unsigned int* c32 = sycl::malloc_shared<unsigned int>(1,q); *c32=0;
  unsigned long long* c64 = sycl::malloc_shared<unsigned long long>(1,q); *c64=0;
  float* fa = sycl::malloc_shared<float>(1,q); *fa=0;
  q.parallel_for(sycl::range<1>(1<<20),[=](sycl::id<1>){ sycl::atomic_ref<unsigned int,sycl::memory_order::relaxed,sycl::memory_scope::device> a(*c32); a.fetch_add(1u);
     sycl::atomic_ref<unsigned long long,sycl::memory_order::relaxed,sycl::memory_scope::device> b(*c64); b.fetch_add(1ull);
     sycl::atomic_ref<float,sycl::memory_order::relaxed,sycl::memory_scope::device> f(*fa); f.fetch_add(1.0f);}).wait();
  std::printf("atomic32 %u atomic64 %llu atomic-float %.0f (expect %d)\n", *c32, *c64, *fa, 1<<20);
  }
  if(T<0||T==5){ // 64-bit CAS (for complex<float> atomic_add emulation as Kokkos does)
  unsigned long long* cas = sycl::malloc_shared<unsigned long long>(1,q); *cas=0;
  q.parallel_for(sycl::range<1>(1<<16),[=](sycl::id<1>){ sycl::atomic_ref<unsigned long long,sycl::memory_order::relaxed,sycl::memory_scope::device> a(*cas);
     unsigned long long old=a.load(); while(!a.compare_exchange_weak(old, old+3ull)){} }).wait();
  std::printf("cas64 %llu (expect %d)\n", *cas, 3*(1<<16));
  }
  if(T<0||T==6){ // timing: axpy bandwidth
  auto t0=std::chrono::steady_clock::now(); for(int r=0;r<20;++r) q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){ y[i]=0.5f*x[i]+y[i];}); q.wait();
  auto t1=std::chrono::steady_clock::now(); double sec=std::chrono::duration<double>(t1-t0).count()/20;
  std::printf("axpy n=2^24: %.3f ms/iter, %.1f GB/s\n", sec*1e3, 3.0*4*n/sec/1e9);
  }
  return 0;
}
