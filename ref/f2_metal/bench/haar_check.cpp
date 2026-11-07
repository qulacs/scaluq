#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <scaluq/all.hpp>
using namespace scaluq;
constexpr Precision P = Precision::F32;
int main(int argc, char** argv) {
    Kokkos::ScopeGuard g(argc, argv);
    std::printf("n  time_ms  norm(dbl)  E[d|a|^2]  E[(d|a|^2)^2](exp:2)  mean_re*sqrt(d)  max|dev-host|  nonfinite\n");
    for (int n : {16, 20, 24, 26, 28}) {
        Kokkos::fence();
        auto t0 = std::chrono::steady_clock::now();
        auto d = StateVector<P, ExecutionSpace::Default>::Haar_random_state(n, 7);
        Kokkos::fence();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        auto a = d.get_amplitudes();
        const double dim = double(a.size());
        double s = 0, m1 = 0, m2 = 0, re = 0; std::size_t bad = 0;
        for (auto& x : a) {
            if (!std::isfinite(x.real()) || !std::isfinite(x.imag())) { ++bad; continue; }
            double p = std::norm(x); s += p; m1 += dim * p; m2 += dim * p * dim * p; re += x.real();
        }
        double maxdev = -1;
        if (n <= 24) {
            auto h = StateVector<P, ExecutionSpace::HostSerial>::Haar_random_state(n, 7).get_amplitudes();
            double sh = 0; for (auto& x : h) sh += std::norm(x);
            maxdev = 0;
            for (std::size_t i = 0; i < a.size(); ++i)
                maxdev = std::max(maxdev, std::abs(a[i] / std::sqrt(s) - h[i] / std::sqrt(sh)) * std::sqrt(dim));
        }
        std::printf("%d  %.2f  %.7f  %.5f  %.5f  %.4f  %.2e  %zu\n", n, ms, s, m1 / dim, m2 / dim,
                    re / std::sqrt(s) , maxdev, bad);
    }
}
