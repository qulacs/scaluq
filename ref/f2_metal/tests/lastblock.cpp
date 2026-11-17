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
