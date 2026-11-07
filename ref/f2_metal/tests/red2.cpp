#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
using C = Kokkos::complex<float>;
int main(int argc,char**argv){ Kokkos::initialize(argc,argv); int bad=0, total=0; {
  for (int rep=0; rep<20; ++rep)
  for (int lg : {8, 9, 12, 16, 20, 22, 24, 26}) {
    long n = 1L<<lg; Kokkos::View<C*> v("v", n);
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(long i){ float a = float((i*2654435761u) % 1000) / 1000.f; v(i) = C(a, 1.f - a); });
    C r; Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(long i, C& s){ s += v(i)*v(i); }, r);  // sum of squares (complex)
    float nr=0; Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(long i, float& s){ s += Kokkos::norm(v(i)); }, nr);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    double rr=0, ri=0, rn=0; for (long i=0;i<n;++i){ double a=h(i).real(), b=h(i).imag(); rr+=a*a-b*b; ri+=2*a*b; rn+=a*a+b*b; }
    double e1 = std::hypot(r.real()-rr, r.imag()-ri)/std::max(1.0, std::hypot(rr,ri)), e2 = std::fabs(nr-rn)/rn;
    ++total; if (e1>1e-4 || e2>1e-4) { ++bad; std::printf("BAD rep=%d n=2^%d e_complex=%.2e e_norm=%.2e\n", rep, lg, e1, e2); }
    if (rep==0) std::printf("n=2^%d rel.err complex %.2e norm %.2e\n", lg, e1, e2);
  }
 } Kokkos::finalize(); std::printf("%d/%d bad\n", bad, total); }
