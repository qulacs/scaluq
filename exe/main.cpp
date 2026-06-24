#include <Kokkos_SIMD.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <scaluq/all.hpp>
#include <string>
#include <type_traits>
#include <vector>

#include "../src/gate/update_ops.hpp"

namespace {
using Space = std::integral_constant<scaluq::ExecutionSpace, scaluq::ExecutionSpace::Default>;

constexpr std::uint64_t n_qubits = 20;
int warmup_iterations = 1000;
int measure_iterations = 10000;

struct Case {
    std::string name;
    std::uint64_t target;
    std::vector<std::uint64_t> controls;
    std::vector<std::uint64_t> control_values;
};

struct Quartiles {
    double q1;
    double median;
    double q3;
};

std::vector<scaluq::StdComplex> make_initial_state() {
    const std::uint64_t dim = 1ULL << n_qubits;
    std::vector<scaluq::StdComplex> amplitudes(dim);
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    for (std::uint64_t i = 0; i < dim; ++i) {
        const double phase = static_cast<double>((i * 104729ULL) % 65536ULL) * 0.0001;
        amplitudes[i] = scale * scaluq::StdComplex(std::cos(phase), std::sin(phase));
    }
    return amplitudes;
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix2x2<Prec> make_internal_matrix() {
    using Complex = scaluq::internal::Complex<Prec>;
    const auto inv_sqrt2 = static_cast<scaluq::internal::Float<Prec>>(1.0 / std::sqrt(2.0));
    return scaluq::internal::Matrix2x2<Prec>{Complex(inv_sqrt2, 0.0),
                                             Complex(0.0, inv_sqrt2),
                                             Complex(0.0, inv_sqrt2),
                                             Complex(inv_sqrt2, 0.0)};
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
                                      scaluq::StateVector<Prec, Space::value>& state) {
    using Complex = scaluq::internal::Complex<Prec>;
    Kokkos::parallel_for(
        "benchmark_baseline_one_target_dense_matrix",
        Kokkos::RangePolicy<scaluq::internal::SpaceType<Space::value>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis0 =
                scaluq::internal::insert_zero_at_mask_positions(it, control_mask | target_mask) |
                control_value_mask;
            std::uint64_t basis1 = basis0 | target_mask;
            Complex val0 = state._raw[basis0];
            Complex val1 = state._raw[basis1];
            Complex res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            state._raw[basis0] = res0;
            state._raw[basis1] = res1;
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
std::vector<double> measure(F&& function) {
    for (int i = 0; i < warmup_iterations; ++i) {
        function();
    }
    Kokkos::fence();

    std::vector<double> samples;
    samples.reserve(measure_iterations);
    for (int i = 0; i < measure_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        function();
        Kokkos::fence();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    return samples;
}

double max_abs_diff(const std::vector<scaluq::StdComplex>& a,
                    const std::vector<scaluq::StdComplex>& b) {
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

template <scaluq::Precision Prec>
void run_case(const Case& bench_case, const std::vector<scaluq::StdComplex>& initial) {
    using State = scaluq::StateVector<Prec, Space::value>;
    const auto internal_matrix = make_internal_matrix<Prec>();

    const std::uint64_t target_mask = 1ULL << bench_case.target;
    const std::uint64_t control_mask = mask_from(bench_case.controls);
    const std::uint64_t control_value_mask =
        value_mask_from(bench_case.controls, bench_case.control_values);

    State baseline_state(n_qubits);
    State implemented_state(n_qubits);
    baseline_state.load(initial);
    implemented_state.load(initial);

    baseline_one_target_dense_matrix<Prec>(
        target_mask, control_mask, control_value_mask, internal_matrix, baseline_state);
    scaluq::internal::one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, internal_matrix, implemented_state);
    Kokkos::fence();

    const auto baseline_amplitudes = baseline_state.get_amplitudes();
    const double implemented_diff =
        max_abs_diff(baseline_amplitudes, implemented_state.get_amplitudes());

    baseline_state.load(initial);
    implemented_state.load(initial);

    auto baseline_samples = measure([&]() {
        baseline_one_target_dense_matrix<Prec>(
            target_mask, control_mask, control_value_mask, internal_matrix, baseline_state);
    });
    auto implemented_samples = measure([&]() {
        scaluq::internal::one_target_dense_matrix_gate(
            target_mask, control_mask, control_value_mask, internal_matrix, implemented_state);
    });

    const auto baseline_q = quartiles(std::move(baseline_samples));
    const auto implemented_q = quartiles(std::move(implemented_samples));

    std::cout << std::left << std::setw(28) << bench_case.name << " target=" << std::setw(2)
              << bench_case.target << " controls=" << format_indices(bench_case.controls) << "\n";
    std::cout << "  baseline ns: q1=" << baseline_q.q1 << " median=" << baseline_q.median
              << " q3=" << baseline_q.q3 << "\n";
    std::cout << "  impl ns:     q1=" << implemented_q.q1 << " median=" << implemented_q.median
              << " q3=" << implemented_q.q3
              << " speedup=" << baseline_q.median / implemented_q.median
              << "x max_abs_diff=" << std::scientific << implemented_diff << std::defaultfloat
              << "\n";
}

template <scaluq::Precision Prec>
void run_precision(const std::vector<Case>& cases,
                   const std::vector<scaluq::StdComplex>& initial,
                   const char* label) {
    using Scalar = std::conditional_t<Prec == scaluq::Precision::F64, double, float>;
    using Simd = Kokkos::Experimental::simd<Scalar>;
    std::cout << "\nprecision=" << label << " simd_lanes=" << Simd::size() << "\n";
    for (const auto& bench_case : cases) {
        run_case<Prec>(bench_case, initial);
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2) {
        warmup_iterations = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        measure_iterations = std::atoi(argv[2]);
    }
    scaluq::initialize();
    {
        std::cout << "n_qubits=" << n_qubits << " warmup=" << warmup_iterations
                  << " samples=" << measure_iterations << "\n";
        const auto initial = make_initial_state();
        const std::vector<Case> cases{
            {"target-0 indexed", 0, {}, {}},
            {"target-4 indexed", 4, {}, {}},
            {"target-8 indexed", 8, {}, {}},
            {"target-4 control-0", 4, {0}, {1}},
            {"target-4 control-8", 4, {8}, {1}},
        };
        run_precision<scaluq::Precision::F64>(cases, initial, "f64");
        run_precision<scaluq::Precision::F32>(cases, initial, "f32");
    }
    scaluq::finalize();
}
