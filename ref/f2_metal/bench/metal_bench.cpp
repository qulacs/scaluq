// Small f32 benchmark of scaluq on the default execution space (Metal via AdaptiveCpp, or
// OpenMP on the CPU). Output: TSV lines "op\tn_qubits\tms_per_op\tGBps\tcheck".
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <scaluq/all.hpp>
#include <vector>

using namespace scaluq;
constexpr Precision Prec = Precision::F32;
constexpr ExecutionSpace Space = ExecutionSpace::Default;
using State = StateVector<Prec, Space>;

template <class F>
double time_ms(int reps, F&& f) {
    for (int i = 0; i < 3; ++i) f();
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    Kokkos::fence();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
               .count() /
           reps;
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] != '-') sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {16, 20, 24, 26, 28};
    std::printf("op\tn_qubits\tms_per_op\tGBps\tcheck\n");
    for (int n : sizes) {
        const double bytes = 8.0 * static_cast<double>(1ULL << n);  // complex<float> state
        const int reps = n <= 20 ? 200 : (n <= 24 ? 50 : 20);
        // Haar init on the host: the device Random_XorShift64_Pool spins on its state locks
        // when dim >> pool size and can leave a runaway kernel on Metal (seen at n=24).
        State state(n);
        state.load(StateVector<Prec, ExecutionSpace::HostSerial>::Haar_random_state(n, 1)
                       .get_amplitudes());
        state.normalize();  // HostSerial f32 normalizes with a serial float sum (~2% off at n=24)
        std::fprintf(stderr, "# n=%d init squared_norm=%.7g\n", n, state.get_squared_norm());
        auto report = [&](const char* op, double ms, double traffic, double check) {
            std::printf(
                "%s\t%d\t%.4f\t%.1f\t%.6g\n", op, n, ms, traffic / (ms * 1e-3) / 1e9, check);
        };
        struct G {
            const char* name;
            Gate<Prec> gate;
        };
        std::vector<G> gates = {{"H(0)", gate::H<Prec>(0)},
                                {"H(n-1)", gate::H<Prec>(n - 1)},
                                {"X(0)", gate::X<Prec>(0)},
                                {"CX(0,n-1)", gate::CX<Prec>(0, n - 1)},
                                {"RX(n/2)", gate::RX<Prec>(n / 2, 0.3)}};
        for (auto& g : gates) {
            double ms = time_ms(reps, [&] { g.gate->update_quantum_state(state); });
            report(g.name, ms, 2.0 * bytes, state.get_squared_norm());
        }
        double norm = 0;
        double ms = time_ms(reps, [&] { norm = state.get_squared_norm(); });
        report("squared_norm", ms, bytes, norm);
        Operator<Prec, Space> op(
            {PauliOperator<Prec>("Z 0 Z 1", 1.0), PauliOperator<Prec>("X 0 X 2", 0.5)});
        StdComplex ev;
        ms = time_ms(reps, [&] { ev = op.get_expectation_value(state); });
        report("expectation(2 terms)", ms, 2.0 * bytes, ev.real());
        std::fflush(stdout);
    }
    return 0;
}
