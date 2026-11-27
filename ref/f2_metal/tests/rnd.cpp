#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdio>
// Same seed must give the same numbers (state assignment must be deterministic).
static auto gen(long n) {
  Kokkos::Random_XorShift64_Pool<> pool(42);
  Kokkos::View<float*> v("v", n);
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(long i) { auto g = pool.get_state(); v(i) = g.frand(); pool.free_state(g); });
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
}
int main(int argc, char** argv) { Kokkos::initialize(argc, argv); int bad = 0; {
  for (long n : {64L, 1000L, 1L << 16, 1L << 20}) for (int rep = 0; rep < 5; ++rep) {
    auto a = gen(n), b = gen(n); long diff = 0; double s = 0;
    for (long i = 0; i < n; ++i) { diff += a(i) != b(i); s += a(i); }
    if (diff) ++bad;
    if (rep == 0 || diff) std::printf("n=%ld rep=%d differing=%ld mean=%.4f\n", n, rep, diff, s / n);
  }
 } Kokkos::finalize(); std::printf("%d bad\n", bad); }
