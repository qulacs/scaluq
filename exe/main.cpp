#include <Kokkos_SIMD.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <scaluq/all.hpp>
#include <scaluq/type/complex.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../src/gate/update_ops.hpp"

namespace {
using Space = std::integral_constant<scaluq::ExecutionSpace, scaluq::ExecutionSpace::Default>;

std::uint64_t min_qubits = 4;
std::uint64_t max_qubits = 22;
int warmup_iterations = 3;
int measure_iterations = 10;
bool include_extra_gates = false;
bool show_quartiles = false;

struct Case {
    std::string name;
    std::uint64_t target;
    std::vector<std::uint64_t> controls;
    std::vector<std::uint64_t> control_values;
};

enum class GateKind { X, Y, H, RX, U3 };

struct Quartiles {
    double q1;
    double median;
    double q3;
};

std::string gate_name(GateKind gate) {
    switch (gate) {
        case GateKind::X:
            return "X";
        case GateKind::Y:
            return "Y";
        case GateKind::H:
            return "H";
        case GateKind::RX:
            return "RX";
        case GateKind::U3:
            return "U3";
    }
    return "unknown";
}

std::string coef_pattern(GateKind gate) {
    switch (gate) {
        case GateKind::X:
            return "offdiag-real";
        case GateKind::Y:
            return "offdiag-imag";
        case GateKind::H:
            return "real-only";
        case GateKind::RX:
            return "diag-real/offdiag-imag";
        case GateKind::U3:
            return "general";
    }
    return "unknown";
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix2x2<Prec> make_x_matrix() {
    using Complex = scaluq::internal::Complex<Prec>;
    return scaluq::internal::Matrix2x2<Prec>{
        Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)};
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix2x2<Prec> make_y_matrix() {
    using Complex = scaluq::internal::Complex<Prec>;
    return scaluq::internal::Matrix2x2<Prec>{
        Complex(0.0, 0.0), Complex(0.0, -1.0), Complex(0.0, 1.0), Complex(0.0, 0.0)};
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix2x2<Prec> make_h_matrix() {
    using Complex = scaluq::internal::Complex<Prec>;
    const auto inv_sqrt2 = static_cast<scaluq::internal::Float<Prec>>(1.0 / std::sqrt(2.0));
    return scaluq::internal::Matrix2x2<Prec>{Complex(inv_sqrt2, 0.0),
                                             Complex(inv_sqrt2, 0.0),
                                             Complex(inv_sqrt2, 0.0),
                                             Complex(-inv_sqrt2, 0.0)};
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix2x2<Prec> make_rx_matrix(double angle) {
    using Complex = scaluq::internal::Complex<Prec>;
    const auto cosval = static_cast<scaluq::internal::Float<Prec>>(std::cos(angle / 2.0));
    const auto sinval = static_cast<scaluq::internal::Float<Prec>>(std::sin(angle / 2.0));
    return scaluq::internal::Matrix2x2<Prec>{
        Complex(cosval, 0.0), Complex(0.0, -sinval), Complex(0.0, -sinval), Complex(cosval, 0.0)};
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix2x2<Prec> make_u3_matrix(double theta, double phi, double lambda) {
    using Complex = scaluq::internal::Complex<Prec>;
    const auto cosval = static_cast<scaluq::internal::Float<Prec>>(std::cos(theta / 2.0));
    const auto sinval = static_cast<scaluq::internal::Float<Prec>>(std::sin(theta / 2.0));
    const auto exp_i_phi = Complex(std::cos(phi), std::sin(phi));
    const auto exp_i_lambda = Complex(std::cos(lambda), std::sin(lambda));
    return scaluq::internal::Matrix2x2<Prec>{Complex(cosval, 0.0),
                                             -exp_i_lambda * sinval,
                                             exp_i_phi * sinval,
                                             exp_i_phi * exp_i_lambda * cosval};
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix2x2<Prec> make_internal_matrix(GateKind gate) {
    switch (gate) {
        case GateKind::X:
            return make_x_matrix<Prec>();
        case GateKind::Y:
            return make_y_matrix<Prec>();
        case GateKind::H:
            return make_h_matrix<Prec>();
        case GateKind::RX:
            return make_rx_matrix<Prec>(0.37);
        case GateKind::U3:
            return make_u3_matrix<Prec>(0.37, 0.37, 0.37);
    }
    return make_h_matrix<Prec>();
}

std::uint64_t mask_from(const std::vector<std::uint64_t>& indices) {
    std::uint64_t mask = 0;
    for (auto index : indices) {
        mask |= 1ULL << index;
    }
    return mask;
}

std::uint64_t value_mask_from(const std::vector<std::uint64_t>& indices,
                              const std::vector<std::uint64_t>& values) {
    std::uint64_t mask = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (values[i] != 0) {
            mask |= 1ULL << indices[i];
        }
    }
    return mask;
}

std::string format_indices(const std::vector<std::uint64_t>& indices) {
    std::string result = "{";
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) {
            result += ",";
        }
        result += std::to_string(indices[i]);
    }
    result += "}";
    return result;
}

template <scaluq::Precision Prec>
void baseline_one_target_dense_matrix(std::uint64_t target_mask,
                                      std::uint64_t control_mask,
                                      std::uint64_t control_value_mask,
                                      const scaluq::internal::Matrix2x2<Prec>& matrix,
                                      auto& state) {
    using Complex = scaluq::internal::Complex<Prec>;
    Kokkos::parallel_for(
        "benchmark_baseline_one_target_dense_matrix",
        Kokkos::RangePolicy<scaluq::internal::SpaceType<Space::value>>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis0 =
                scaluq::internal::insert_zero_at_mask_positions(it, control_mask | target_mask) |
                control_value_mask;
            std::uint64_t basis1 = basis0 | target_mask;
            Complex val0 = state.at_unsafe(basis0);
            Complex val1 = state.at_unsafe(basis1);
            Complex res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            state.at_unsafe(basis0) = res0;
            state.at_unsafe(basis1) = res1;
        });
}

Quartiles quartiles(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    auto pick = [&](double p) {
        const double pos = p * static_cast<double>(samples.size() - 1);
        const auto lo = static_cast<std::size_t>(std::floor(pos));
        const auto hi = static_cast<std::size_t>(std::ceil(pos));
        const double frac = pos - static_cast<double>(lo);
        return samples[lo] * (1.0 - frac) + samples[hi] * frac;
    };
    return {pick(0.25), pick(0.50), pick(0.75)};
}

template <typename F>
std::vector<double> measure(F&& function, int iterations) {
    for (int i = 0; i < warmup_iterations; ++i) {
        function();
    }
    Kokkos::fence();

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        function();
        Kokkos::fence();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    return samples;
}

template <typename F>
std::vector<double> measure(F&& function) {
    return measure(std::forward<F>(function), measure_iterations);
}

double max_abs_diff(const std::vector<scaluq::StdComplex>& a,
                    const std::vector<scaluq::StdComplex>& b) {
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

void print_row(GateKind gate,
               const Case& bench_case,
               const char* path,
               const Quartiles& baseline_q,
               const Quartiles& simd_q,
               double simd_diff) {
    std::cout << "| " << std::left << std::setw(2) << gate_name(gate) << " | " << std::setw(22)
              << coef_pattern(gate) << " | " << std::setw(18) << bench_case.name << " | "
              << std::right << std::setw(2) << bench_case.target << " | " << std::setw(9)
              << format_indices(bench_case.controls) << " | " << std::left << std::setw(17) << path
              << " | " << std::right;
    if (show_quartiles) {
        std::cout << std::setw(12) << baseline_q.q1 << " | ";
    }
    std::cout << std::setw(13) << baseline_q.median << " | ";
    if (show_quartiles) {
        std::cout << std::setw(12) << baseline_q.q3 << " | " << std::setw(10) << simd_q.q1 << " | ";
    }
    std::cout << std::setw(11) << simd_q.median << " | ";
    if (show_quartiles) {
        std::cout << std::setw(10) << simd_q.q3 << " | ";
    }
    std::cout << std::setw(8) << baseline_q.median / simd_q.median << " | " << std::scientific
              << std::setw(11) << simd_diff << std::defaultfloat << " |\n";
}

template <scaluq::Precision Prec>
void run_case(GateKind gate,
              const Case& bench_case,
              const scaluq::StateVector<Prec, Space::value>& initial,
              int iterations) {
    using State = scaluq::StateVector<Prec, Space::value>;
    const auto internal_matrix = make_internal_matrix<Prec>(gate);

    const std::uint64_t target_mask = 1ULL << bench_case.target;
    const std::uint64_t control_mask = mask_from(bench_case.controls);
    const std::uint64_t control_value_mask =
        value_mask_from(bench_case.controls, bench_case.control_values);
    constexpr std::size_t complex_lanes =
        scaluq::internal::SimdComplex<Prec>::complex_lanes;
    const std::uint64_t inlane_mask = complex_lanes - 1;
    const char* path = (target_mask & inlane_mask) != 0
                           ? ((control_mask & inlane_mask) == 0 ? "in-lane" : "scalar")
                           : ((control_mask & inlane_mask) == 0 ? "contiguous" : "scalar");

    State baseline_state = initial.copy();
    State simd_state = initial.copy();

    baseline_one_target_dense_matrix<Prec>(
        target_mask, control_mask, control_value_mask, internal_matrix, baseline_state);
    scaluq::internal::one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, internal_matrix, simd_state);
    Kokkos::fence();

    const auto baseline_amplitudes = baseline_state.get_amplitudes();
    const double simd_diff = max_abs_diff(baseline_amplitudes, simd_state.get_amplitudes());

    baseline_state.load(initial);
    simd_state.load(initial);

    auto baseline_samples = measure(
        [&]() {
            baseline_one_target_dense_matrix<Prec>(
                target_mask, control_mask, control_value_mask, internal_matrix, baseline_state);
        },
        iterations);
    auto simd_samples = measure(
        [&]() {
            scaluq::internal::one_target_dense_matrix_gate(
                target_mask, control_mask, control_value_mask, internal_matrix, simd_state);
        },
        iterations);

    const auto baseline_q = quartiles(std::move(baseline_samples));
    const auto simd_q = quartiles(std::move(simd_samples));

    print_row(gate, bench_case, path, baseline_q, simd_q, simd_diff);
}

void print_table_header() {
    std::cout << "| gate | coef pattern          | case               | target | controls  | path              | ";
    if (show_quartiles) {
        std::cout << "scalar q1 ns | ";
    }
    std::cout << "scalar med ns | ";
    if (show_quartiles) {
        std::cout << "scalar q3 ns | simd q1 ns | ";
    }
    std::cout << "simd med ns | ";
    if (show_quartiles) {
        std::cout << "simd q3 ns | ";
    }
    std::cout << "speedup | max diff    |\n";

    std::cout << "| ---  | ---                   | ---                | ---:   | ---       | ---               | ";
    if (show_quartiles) {
        std::cout << "---:         | ";
    }
    std::cout << "---:          | ";
    if (show_quartiles) {
        std::cout << "---:         | ---:       | ";
    }
    std::cout << "---:        | ";
    if (show_quartiles) {
        std::cout << "---:       | ";
    }
    std::cout << "---:    | ---:        |\n";
}

template <scaluq::Precision Prec>
void run_precision(std::uint64_t n_qubits, const std::vector<Case>& cases, const char* label) {
    using Scalar = std::conditional_t<Prec == scaluq::Precision::F64, double, float>;
    using Simd = Kokkos::Experimental::simd<Scalar>;
    constexpr std::uint64_t seed = 314159;

    const auto initial = scaluq::StateVector<Prec, Space::value>::Haar_random_state(n_qubits, seed);
    std::cout << "\nstate=StateVector precision=" << label << " simd_lanes=" << Simd::size()
              << " n_qubits=" << n_qubits << " samples=" << measure_iterations << "\n";
    print_table_header();
    const std::vector<GateKind> gates =
        include_extra_gates
            ? std::vector<GateKind>{
                  GateKind::X, GateKind::Y, GateKind::H, GateKind::RX, GateKind::U3}
            : std::vector<GateKind>{GateKind::X, GateKind::Y, GateKind::H, GateKind::RX};
    for (const auto gate : gates) {
        for (const auto& bench_case : cases) {
            run_case<Prec>(gate, bench_case, initial, measure_iterations);
        }
    }

}
}  // namespace

int main(int argc, char** argv) {
    int positional_count = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--all-gates") {
            include_extra_gates = true;
        } else if (argument == "--quartiles") {
            show_quartiles = true;
        } else if (argument == "--min-qubits" && i + 1 < argc) {
            min_qubits = std::strtoull(argv[++i], nullptr, 10);
        } else if (argument == "--max-qubits" && i + 1 < argc) {
            max_qubits = std::strtoull(argv[++i], nullptr, 10);
        } else if (positional_count == 0) {
            warmup_iterations = std::atoi(argv[i]);
            ++positional_count;
        } else if (positional_count == 1) {
            measure_iterations = std::atoi(argv[i]);
            ++positional_count;
        }
    }
    scaluq::initialize();
    {
        std::cout << "n_qubits=" << min_qubits << ".." << max_qubits
                  << " warmup=" << warmup_iterations
                  << " samples=" << measure_iterations
                  << " gates=" << (include_extra_gates ? "X,Y,H,RX,U3" : "X,Y,H,RX")
                  << " quartiles=" << (show_quartiles ? "on" : "off") << "\n";
        for (std::uint64_t n_qubits = min_qubits; n_qubits <= max_qubits; ++n_qubits) {
            std::vector<Case> cases{{"target-0 in-lane", 0, {}, {}}};
            for (std::uint64_t target = 4; target < n_qubits; target += 4) {
                cases.push_back(
                    {"target-" + std::to_string(target) + " indexed", target, {}, {}});
            }
            if (n_qubits > 4) {
                cases.push_back({"target-0 control-4", 0, {4}, {1}});
                cases.push_back({"target-4 control-0", 4, {0}, {1}});
            }
            if (n_qubits > 8) {
                cases.push_back({"target-4 control-8", 4, {8}, {1}});
            }
            run_precision<scaluq::Precision::F64>(n_qubits, cases, "f64");
            run_precision<scaluq::Precision::F32>(n_qubits, cases, "f32");
        }
    }
    scaluq::finalize();
}
