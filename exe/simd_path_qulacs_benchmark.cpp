#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <scaluq/all.hpp>

#include "../src/gate/update_ops.hpp"
#include "../src/gate/update_ops_matrix_4x4.hpp"
#include "csim/type.hpp"
#include "csim/update_ops.hpp"

namespace {

struct Options {
    std::uint64_t min_qubits = 4;
    std::uint64_t max_qubits = 24;
    int warmups = 5;
    int iterations = 20;
    std::string output = "benchmark-results/simd-path-comparison.csv";
};

struct PathCase {
    const char* name;
    std::uint64_t target0;
    std::uint64_t target1;
    bool supports_f64;
};

struct Samples {
    std::vector<double> scaluq_us;
    std::vector<double> qulacs_us;
};

constexpr PathCase path_cases[] = {
    {"low", 0, 1, false},
    {"middle", 0, 2, true},
    {"high", 2, 3, true},
};

std::uint64_t parse_u64(const char* text, std::string_view option) {
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::runtime_error("invalid value for " + std::string(option) + ": " + text);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&]() -> const char* {
            if (++i >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[i];
        };
        if (argument == "--min-qubits") {
            options.min_qubits = parse_u64(value(), argument);
        } else if (argument == "--max-qubits") {
            options.max_qubits = parse_u64(value(), argument);
        } else if (argument == "--warmup") {
            options.warmups = static_cast<int>(parse_u64(value(), argument));
        } else if (argument == "--iterations") {
            options.iterations = static_cast<int>(parse_u64(value(), argument));
        } else if (argument == "--output") {
            options.output = value();
        } else if (argument == "--help") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --min-qubits N   first qubit count (default: 4)\n"
                << "  --max-qubits N   last qubit count (default: 24)\n"
                << "  --warmup N       alternating warm-up pairs (default: 5)\n"
                << "  --iterations N   measured pairs; positive and even (default: 20)\n"
                << "  --output PATH    output CSV\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.min_qubits < 4 || options.min_qubits > options.max_qubits ||
        options.warmups < 0 || options.iterations <= 0 || (options.iterations & 1) != 0) {
        throw std::runtime_error(
            "require 4 <= min-qubits <= max-qubits and a positive, even iteration count");
    }
    return options;
}

std::vector<CTYPE> make_qulacs_matrix() {
    // A unitary 4-point DFT keeps repeated timed updates numerically bounded.
    const CTYPE one{0.5, 0.0};
    const CTYPE minus_one{-0.5, 0.0};
    const CTYPE imag{0.0, 0.5};
    const CTYPE minus_imag{0.0, -0.5};
    return {
        one, one, one, one,
        one, imag, minus_one, minus_imag,
        one, minus_one, one, minus_one,
        one, minus_imag, minus_one, imag,
    };
}

template <scaluq::Precision Prec>
scaluq::internal::Matrix4x4<Prec> make_scaluq_matrix(const std::vector<CTYPE>& source) {
    using Complex = scaluq::internal::Complex<Prec>;
    scaluq::internal::Matrix4x4<Prec> matrix;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            const auto value = source[row * 4 + col];
            matrix[row][col] = Complex(value.real(), value.imag());
        }
    }
    return matrix;
}

std::vector<CTYPE> make_initial_state(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    std::vector<CTYPE> state(dim);
    double norm2 = 0.0;
    for (std::uint64_t i = 0; i < dim; ++i) {
        const double real = std::sin(static_cast<double>(i + 1) * 0.17);
        const double imag = std::cos(static_cast<double>(i + 1) * 0.11);
        state[i] = {real, imag};
        norm2 += std::norm(state[i]);
    }
    const double inv_norm = 1.0 / std::sqrt(norm2);
    for (auto& value : state) value *= inv_norm;
    return state;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() & 1U) return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

template <class Function>
double time_us(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

template <class ScaluqUpdate, class QulacsUpdate>
Samples measure_alternating(const Options& options,
                            ScaluqUpdate&& update_scaluq,
                            QulacsUpdate&& update_qulacs) {
    auto run_scaluq = [&] {
        const auto elapsed = time_us([&] {
            update_scaluq();
            Kokkos::fence();
        });
        return elapsed;
    };
    auto run_qulacs = [&] { return time_us(update_qulacs); };

    for (int i = 0; i < options.warmups; ++i) {
        if ((i & 1) == 0) {
            run_scaluq();
            run_qulacs();
        } else {
            run_qulacs();
            run_scaluq();
        }
    }

    Samples samples;
    samples.scaluq_us.reserve(options.iterations);
    samples.qulacs_us.reserve(options.iterations);
    for (int i = 0; i < options.iterations; ++i) {
        if (((options.warmups + i) & 1) == 0) {
            samples.scaluq_us.push_back(run_scaluq());
            samples.qulacs_us.push_back(run_qulacs());
        } else {
            samples.qulacs_us.push_back(run_qulacs());
            samples.scaluq_us.push_back(run_scaluq());
        }
    }
    return samples;
}

template <scaluq::Precision Prec, scaluq::ExecutionSpace Space>
void benchmark_case(std::ofstream& csv,
                    const Options& options,
                    const PathCase& path,
                    std::uint64_t n_qubits,
                    const std::vector<CTYPE>& qulacs_matrix) {
    using State = scaluq::StateVector<Prec, Space>;
    const auto initial = make_initial_state(n_qubits);
    State scaluq_state(n_qubits);
    scaluq_state.load(std::vector<scaluq::StdComplex>(initial.begin(), initial.end()));
    std::vector<CTYPE> qulacs_state = initial;
    const auto scaluq_matrix = make_scaluq_matrix<Prec>(qulacs_matrix);
    const std::uint64_t target_mask =
        (1ULL << path.target0) | (1ULL << path.target1);

    auto update_scaluq = [&] {
        scaluq::internal::two_target_dense_matrix_gate(
            target_mask, 0, 0, scaluq_matrix, scaluq_state);
    };
    auto update_qulacs = [&] {
        double_qubit_dense_matrix_gate_c(path.target0,
                                         path.target1,
                                         qulacs_matrix.data(),
                                         qulacs_state.data(),
                                         qulacs_state.size());
    };

    update_scaluq();
    Kokkos::fence();
    update_qulacs();
    const auto scaluq_values = scaluq_state.get_amplitudes();
    double max_error = 0.0;
    for (std::size_t i = 0; i < initial.size(); ++i) {
        max_error = std::max(max_error, std::abs(scaluq_values[i] - qulacs_state[i]));
    }
    const double tolerance = Prec == scaluq::Precision::F32 ? 2e-6 : 1e-12;
    if (max_error > tolerance) {
        throw std::runtime_error("state mismatch in " + std::string(path.name));
    }

    scaluq_state.load(std::vector<scaluq::StdComplex>(initial.begin(), initial.end()));
    qulacs_state = initial;
    const Samples samples = measure_alternating(options, update_scaluq, update_qulacs);
    const double scaluq_median = median(samples.scaluq_us);
    const double qulacs_median = median(samples.qulacs_us);
    const char* precision = Prec == scaluq::Precision::F32 ? "f32" : "f64";

    csv << n_qubits << ',' << path.name << ',' << precision << ",\"{" << path.target0 << ';'
        << path.target1 << "}\"," << scaluq_median << ',' << qulacs_median << ','
        << qulacs_median / scaluq_median << ',' << max_error << '\n';
    std::cout << std::setw(6) << path.name << " " << precision << " q=" << std::setw(2)
              << n_qubits << " Scaluq=" << std::setw(11) << scaluq_median
              << " us Qulacs=" << std::setw(11) << qulacs_median
              << " us speedup=" << qulacs_median / scaluq_median << "x\n";
}

template <scaluq::Precision Prec>
void dispatch_space(std::ofstream& csv,
                    const Options& options,
                    const PathCase& path,
                    std::uint64_t n_qubits,
                    const std::vector<CTYPE>& matrix) {
    if (n_qubits < 13) {
        benchmark_case<Prec, scaluq::ExecutionSpace::HostSerial>(
            csv, options, path, n_qubits, matrix);
    } else {
        benchmark_case<Prec, scaluq::ExecutionSpace::Default>(
            csv, options, path, n_qubits, matrix);
    }
}

void run(const Options& options) {
#if !defined(KOKKOS_ARCH_AVX2) && !defined(KOKKOS_ARCH_AVX512XEON)
    throw std::runtime_error("Scaluq was not configured with an x86 SIMD architecture");
#endif
    std::ofstream csv(options.output);
    if (!csv) throw std::runtime_error("cannot open output CSV: " + options.output);
    csv << std::setprecision(17);
    csv << "qubits,path,precision,targets,scaluq_median_us,qulacs_median_us,"
           "speedup,max_error\n";
    const auto matrix = make_qulacs_matrix();

    std::cout << "\n[f32]\n";
    for (const auto& path : path_cases) {
        std::cout << "\n[" << path.name << " targets={" << path.target0 << ',' << path.target1
                  << "}]\n";
        for (std::uint64_t n_qubits = options.min_qubits; n_qubits <= options.max_qubits;
             ++n_qubits) {
            dispatch_space<scaluq::Precision::F32>(csv, options, path, n_qubits, matrix);
        }
    }
    std::cout << "\n[f64]\n";
    for (const auto& path : path_cases) {
        if (!path.supports_f64) continue;
        std::cout << "\n[" << path.name << " targets={" << path.target0 << ',' << path.target1
                  << "}]\n";
        for (std::uint64_t n_qubits = options.min_qubits; n_qubits <= options.max_qubits;
             ++n_qubits) {
            dispatch_space<scaluq::Precision::F64>(csv, options, path, n_qubits, matrix);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        scaluq::initialize();
        run(options);
        scaluq::finalize();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
