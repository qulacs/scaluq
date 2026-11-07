// Repro for two issues, using only the public API on the default (OpenMP) space.
// 1) Operator::get_expectation_value vs the same terms evaluated one PauliOperator at a time.
// 2) Squared norm accuracy of f32 states: library value vs a double sum of the amplitudes.
#include <Kokkos_Core.hpp>
#include <chrono>
#include <complex>
#include <cstdio>
#include <random>
#include <scaluq/all.hpp>
using namespace scaluq;

template <class F>
double time_ms(int reps, F&& f) {
    f();
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    Kokkos::fence();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
               .count() / reps;
}

template <Precision P>
void expectation(int n, const char* name) {
    constexpr ExecutionSpace S = ExecutionSpace::Default;
    auto state = StateVector<P, S>::Haar_random_state(n, 1);
    PauliOperator<P> p1("Z 0 Z 1", 1.0), p2("X 0 X 2", 0.5);
    Operator<P, S> op({p1, p2});
    StdComplex a, b;
    double t_op = time_ms(5, [&] { a = op.get_expectation_value(state); });
    double t_pauli = time_ms(5, [&] {
        b = p1.get_expectation_value(state) + p2.get_expectation_value(state);
    });
    double norm = 0;
    double t_norm = time_ms(5, [&] { norm = state.get_squared_norm(); });
    std::printf("%s n=%d  Operator(2 terms) %.2f ms | 2x PauliOperator %.2f ms | squared_norm %.2f ms"
                "  (values %.6g / %.6g)\n",
                name, n, t_op, t_pauli, t_norm, a.real(), b.real());
}

template <Precision P, ExecutionSpace S>
void norm_accuracy(int n, const char* name) {
    StateVector<P, S> state(n);
    std::vector<StdComplex> amps(1ULL << n, StdComplex(1.0, 0.0));  // squared norm = 2^n
    state.load(amps);
    double exact = 0;
    for (auto& a : state.get_amplitudes()) exact += std::norm(a);
    double lib = state.get_squared_norm();
    state.normalize();
    double after = 0;
    for (auto& a : state.get_amplitudes()) after += std::norm(a);
    std::printf("%s n=%d  get_squared_norm rel.err %.2e | squared norm after normalize() = %.7f\n",
                name, n, lib / exact - 1, after);
}


template <Precision P, ExecutionSpace S>
void norm_accuracy_random(int n, const char* name) {
    // A normalized random state (normalized in double on the host), and Haar_random_state.
    std::mt19937_64 rng(1);
    std::normal_distribution<double> g;
    std::vector<StdComplex> amps(1ULL << n);
    double s = 0;
    for (auto& a : amps) { a = {g(rng), g(rng)}; s += std::norm(a); }
    for (auto& a : amps) a /= std::sqrt(s);
    StateVector<P, S> state(n);
    state.load(amps);
    auto haar = StateVector<P, S>::Haar_random_state(n, 1);
    double haar_exact = 0;
    for (auto& a : haar.get_amplitudes()) haar_exact += std::norm(a);
    std::printf("%s n=%d  normalized state: get_squared_norm = %.7f | Haar_random_state: exact squared norm = %.7f\n",
                name, n, state.get_squared_norm(), haar_exact);
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard g(argc, argv);
    std::printf("== 1) expectation value, default space, %d threads\n",
                Kokkos::DefaultExecutionSpace().concurrency());
    for (int n : {20, 24, 26}) {
        expectation<Precision::F64>(n, "f64");
        expectation<Precision::F32>(n, "f32");
    }
    if (argc > 1) {
        std::printf("== 2a) squared norm accuracy, normalized random states\n");
        for (int n : {20, 22, 24, 26}) {
            norm_accuracy_random<Precision::F32, ExecutionSpace::HostSerial>(n, "f32 HostSerial");
            norm_accuracy_random<Precision::F32, ExecutionSpace::Default>(n, "f32 Default   ");
        }
        return 0;
    }
    std::printf("== 2) squared norm accuracy, all amplitudes 1\n");
    for (int n : {20, 24, 26}) {
        norm_accuracy<Precision::F32, ExecutionSpace::HostSerial>(n, "f32 HostSerial");
        norm_accuracy<Precision::F32, ExecutionSpace::Default>(n, "f32 Default   ");
        norm_accuracy<Precision::F64, ExecutionSpace::HostSerial>(n, "f64 HostSerial");
    }
}
