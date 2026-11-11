#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){ setvbuf(stdout,nullptr,_IONBF,0); int m=atoi(argv[1]);
  sycl::queue q{sycl::gpu_selector_v}; const size_t n=1024, wg=256;
  float* in=sycl::malloc_shared<float>(n,q); float* out=sycl::malloc_shared<float>(n,q); unsigned* info=sycl::malloc_shared<unsigned>(8,q);
  for(size_t i=0;i<n;++i){in[i]=1.0f; out[i]=-1;}
  q.parallel_for(sycl::nd_range<1>(n,wg),[=](sycl::nd_item<1> it){ auto sg=it.get_sub_group(); size_t g=it.get_global_id(0); float v=in[g];
    if(g==0){ info[0]=sg.get_local_range()[0]; info[1]=sg.get_max_local_range()[0]; info[2]=sg.get_group_range()[0]; }
    if(m==0) out[g]=sycl::reduce_over_group(sg,v,sycl::plus<float>());
    if(m==1){ float s=v; for(unsigned o=1;o<sg.get_local_range()[0];o<<=1) s+= sycl::shift_group_left(sg,s,o) * ((sg.get_local_id()[0]+o<sg.get_local_range()[0])?1.f:0.f); out[g]=s; }
    if(m==2){ float s=v; for(unsigned o=sg.get_local_range()[0]/2;o>0;o>>=1) s+= sycl::permute_group_by_xor(sg,s,o); out[g]=s; }
    if(m==3){ out[g]=sycl::select_from_group(sg, (float)sg.get_local_id()[0], (sg.get_local_id()[0]+1)%32); }
    if(m==4){ out[g]=sycl::group_broadcast(sg, (float)sg.get_local_id()[0]+v, 5); }
    if(m==5){ out[g]=sycl::shift_group_right(sg, (float)sg.get_local_id()[0], 1); }
  }).wait();
  std::printf("mode %d: sg local_range=%u max=%u num_sg=%u | out[0..3]=%g %g %g %g out[31]=%g out[32]=%g out[255]=%g\n", m, info[0],info[1],info[2], out[0],out[1],out[2],out[3],out[31],out[32],out[255]);
}
