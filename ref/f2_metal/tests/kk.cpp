#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
using C = Kokkos::complex<float>;
using Exec = Kokkos::DefaultExecutionSpace;
static double now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  int T = argc>1 ? atoi(argv[1]) : -1;
  Kokkos::initialize(argc, argv);
  {
    std::printf("exec space: %s\n", Exec::name());
    const int n = 1<<20;
    Kokkos::View<float*> x("x", n);
    Kokkos::View<C*> s("s", n);
    Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(int i){ x(i) = float(i%7); s(i) = C(float(i%3), 1.0f); });
    Kokkos::fence(); std::printf("[1] parallel_for ok\n");
    if(T<0||T==2){ float r=0; Kokkos::parallel_reduce("sum", n, KOKKOS_LAMBDA(int i, float& a){ a += x(i); }, r); Kokkos::fence();
      double ref=0; for(int i=0;i<n;++i) ref+=i%7; std::printf("[2] reduce float: %.6e ref %.6e\n", r, ref); }
    if(T<0||T==3){ C r; Kokkos::parallel_reduce("csum", n, KOKKOS_LAMBDA(int i, C& a){ a += s(i); }, r); Kokkos::fence();
      std::printf("[3] reduce complex<float>: (%.6e, %.6e) ref (%.6e, %.6e)\n", r.real(), r.imag(), 1048575.0, (double)n); }
    if(T<0||T==4){ float r=0; Kokkos::parallel_reduce("md", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{1024,1024}), KOKKOS_LAMBDA(int i,int j, float& a){ a += x(i*1024+j); }, r); Kokkos::fence(); std::printf("[4] MDRange reduce: %.6e\n", r); }
    if(T<0||T==5){ using P=Kokkos::TeamPolicy<>; float r=0;
      Kokkos::parallel_reduce("team", P(1024, (argc>2? atoi(argv[2]) : 256)), KOKKOS_LAMBDA(const P::member_type& m, float& a){ float t=0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(m,1024), [&](int j, float& b){ b += x(m.league_rank()*1024+j); }, t);
        Kokkos::single(Kokkos::PerTeam(m), [&]{ a += t; }); }, r); Kokkos::fence(); std::printf("[5] TeamPolicy nested reduce: %.6e\n", r); }
    if(T<0||T==6){ Kokkos::View<float*> y("y", n); float tot=0;
      Kokkos::parallel_scan("scan", n, KOKKOS_LAMBDA(int i, float& u, bool fin){ if(fin) y(i)=u; u += x(i); }, tot); Kokkos::fence();
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y); std::printf("[6] parallel_scan total %.6e y[n-1]=%.6e\n", tot, h(n-1)); }
    if(T<0||T==7){ Kokkos::View<C> a("a"); int m = argc>2? atoi(argv[2]) : 1024;
      double t0=now(); Kokkos::parallel_for("atomic", m, KOKKOS_LAMBDA(int){ Kokkos::atomic_add(&a(), C(1.0f, 2.0f)); }); Kokkos::fence();
      auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a); std::printf("[7] atomic_add complex<float> x%d: (%g,%g) %.1f ms\n", m, h().real(), h().imag(), (now()-t0)*1e3); }
    if(T<0||T==8){ Kokkos::Random_XorShift64_Pool<> pool(1234); Kokkos::View<float*> r("r", 1<<16);
      Kokkos::parallel_for("rand", 1<<16, KOKKOS_LAMBDA(int i){ auto g=pool.get_state(); r(i)=g.frand(0.f,1.f); pool.free_state(g); }); Kokkos::fence();
      auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r); double m=0; for(int i=0;i<(1<<16);++i) m+=h(i); std::printf("[8] Random frand mean %.4f\n", m/(1<<16)); }
    if(T<0||T==9){ Kokkos::View<float*> y("y", 1<<24), z("z", 1<<24); Kokkos::deep_copy(y, 1.0f); Kokkos::deep_copy(z, 2.0f);
      for(int w=0; w<3; ++w) Kokkos::parallel_for("axpy", 1<<24, KOKKOS_LAMBDA(int i){ z(i) = 0.5f*y(i) + z(i); }); Kokkos::fence();
      double t0=now(); for(int w=0; w<50; ++w) Kokkos::parallel_for("axpy", 1<<24, KOKKOS_LAMBDA(int i){ z(i) = 0.5f*y(i) + z(i); }); Kokkos::fence();
      double dt=(now()-t0)/50; std::printf("[9] axpy 2^24 f32: %.3f ms, %.1f GB/s\n", dt*1e3, 3.0*4*(1<<24)/dt/1e9); }
  }
  Kokkos::finalize();
}
