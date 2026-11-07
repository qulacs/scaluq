#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
// Exclusive parallel_scan check: 64-bit ints are exact; float compared with tolerance.
int main(int argc,char**argv){ Kokkos::initialize(argc,argv); int bad=0,total=0; {
  for (int rep=0; rep<10; ++rep)
  for (int lg : {5, 8, 9, 12, 16, 17, 20, 22, 24}) {
    long n = 1L<<lg; if (lg==17) n += 12345;
    Kokkos::View<long*> out("out", n); Kokkos::View<float*> fout("fout", n);
    long tot=0; float ftot=0;
    Kokkos::parallel_scan(n, KOKKOS_LAMBDA(long i, long& upd, bool fin){ if (fin) out(i)=upd; upd += (i*2654435761u)%7; }, tot);
    Kokkos::parallel_scan(n, KOKKOS_LAMBDA(long i, float& upd, bool fin){ if (fin) fout(i)=upd; upd += 1.0f/n; }, ftot);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    auto fh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fout);
    long acc=0, first_bad=-1; for (long i=0;i<n;++i){ if (h(i)!=acc && first_bad<0) first_bad=i; acc += (i*2654435761u)%7; }
    double ferr=0; for (long i=0;i<n;++i) ferr=std::max(ferr, std::fabs(fh(i) - double(i)/n));
    bool ok = first_bad<0 && tot==acc && ferr<1e-4 && std::fabs(ftot-1.0)<1e-4;
    ++total; if(!ok){ ++bad; std::printf("BAD rep=%d n=%ld first_bad=%ld tot=%ld/%ld ferr=%.2e ftot=%.6f\n",rep,n,first_bad,tot,acc,ferr,ftot);}
    if (rep==0) std::printf("n=%ld int ok=%d float maxerr=%.2e total=%.6f\n", n, first_bad<0 && tot==acc, ferr, ftot);
  }
 } Kokkos::finalize(); std::printf("%d/%d bad\n",bad,total); }
