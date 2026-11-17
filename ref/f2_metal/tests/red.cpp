#include <Kokkos_Core.hpp>
#include <cstdio>
int main(int argc,char**argv){ Kokkos::initialize(argc,argv); {
  for (int n : {32, 64, 128, 256, 257, 512, 1024, 4096, 65536, 1<<20}) {
    float r=0; Kokkos::parallel_reduce("r", n, KOKKOS_LAMBDA(int, float& a){ a += 1.0f; }, r);
    int ri=0; Kokkos::parallel_reduce("ri", n, KOKKOS_LAMBDA(int, int& a){ a += 1; }, ri);
    std::printf("n=%d float=%g int=%d\n", n, r, ri);
  }
 } Kokkos::finalize(); }
