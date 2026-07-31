#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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
    std::uint64_t parallel_nqubit_threshold = 13;
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

struct CaseResult {
    double scaluq_ascending_us = 0.0;
    double scaluq_descending_us = 0.0;
    double qulacs_ascending_us = 0.0;
    double qulacs_descending_us = 0.0;
    double max_error = 0.0;
};

struct QulacsState {
    struct Free {
        void operator()(CTYPE* pointer) const { std::free(pointer); }
    };

    std::unique_ptr<CTYPE, Free> storage;
    std::uint64_t dim;

    CTYPE* data() { return storage.get(); }
    const CTYPE& operator[](std::uint64_t index) const { return storage.get()[index]; }
    std::uint64_t size() const { return dim; }
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
    if (const char* threshold = std::getenv("QULACS_PARALLEL_NQUBIT_THRESHOLD")) {
        options.parallel_nqubit_threshold =
            parse_u64(threshold, "QULACS_PARALLEL_NQUBIT_THRESHOLD");
        if (options.parallel_nqubit_threshold == 0 ||
            options.parallel_nqubit_threshold > 64) {
            throw std::runtime_error(
                "QULACS_PARALLEL_NQUBIT_THRESHOLD must be between 1 and 64");
        }
    }
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
                << "  --warmup N       warm-up updates per case (default: 5)\n"
                << "  --iterations N   measured updates per case (default: 20)\n"
                << "  --output PATH    output CSV\n"
                << "Environment:\n"
                << "  QULACS_PARALLEL_NQUBIT_THRESHOLD=N\n"
                << "                    shared Scaluq/Qulacs parallel threshold (default: 13)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.min_qubits < 4 || options.min_qubits > options.max_qubits ||
        options.warmups < 0 || options.iterations <= 0) {
        throw std::runtime_error(
            "require 4 <= min-qubits <= max-qubits and a positive iteration count");
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

QulacsState make_qulacs_state_parallel(const std::vector<CTYPE>& initial) {
    auto* raw = static_cast<CTYPE*>(std::malloc(initial.size() * sizeof(CTYPE)));
    if (raw == nullptr) throw std::bad_alloc();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::uint64_t i = 0; i < initial.size(); ++i) {
        raw[i] = initial[i];
    }
    return {std::unique_ptr<CTYPE, QulacsState::Free>(raw), initial.size()};
}

template <class Function>
double time_us(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

template <class Function>
double measure_one(const Options& options, Function&& function, bool needs_fence) {
    for (int i = 0; i < options.warmups; ++i) {
        function();
        if (needs_fence) Kokkos::fence();
    }

    std::vector<double> samples;
    samples.reserve(options.iterations);
    for (int i = 0; i < options.iterations; ++i) {
        samples.push_back(time_us([&] {
            function();
            if (needs_fence) Kokkos::fence();
        }));
    }
    return median(std::move(samples));
}

template <scaluq::Precision Prec, scaluq::ExecutionSpace Space>
double measure_scaluq_case(const Options& options,
                           const PathCase& path,
                           std::uint64_t n_qubits,
                           const std::vector<CTYPE>& qulacs_matrix) {
    using State = scaluq::StateVector<Prec, Space>;
    const auto initial = make_initial_state(n_qubits);
    State scaluq_state(n_qubits);
    scaluq_state.load(std::vector<scaluq::StdComplex>(initial.begin(), initial.end()));
    const auto scaluq_matrix = make_scaluq_matrix<Prec>(qulacs_matrix);
    const std::uint64_t target_mask = (1ULL << path.target0) | (1ULL << path.target1);

    auto update_scaluq = [&] {
        scaluq::internal::two_target_dense_matrix_gate(
            target_mask, 0, 0, scaluq_matrix, scaluq_state);
    };
    return measure_one(options, update_scaluq, true);
}

double measure_qulacs_case(const Options& options,
                           const PathCase& path,
                           std::uint64_t n_qubits,
                           const std::vector<CTYPE>& qulacs_matrix) {
    const auto initial = make_initial_state(n_qubits);
    auto qulacs_state = make_qulacs_state_parallel(initial);
    auto update_qulacs = [&] {
        double_qubit_dense_matrix_gate_c(path.target0,
                                         path.target1,
                                         qulacs_matrix.data(),
                                         qulacs_state.data(),
                                         qulacs_state.size());
    };
    return measure_one(options, update_qulacs, false);
}

template <scaluq::Precision Prec, scaluq::ExecutionSpace Space>
double validate_case(const PathCase& path,
                     std::uint64_t n_qubits,
                     const std::vector<CTYPE>& qulacs_matrix) {
    using State = scaluq::StateVector<Prec, Space>;
    const auto initial = make_initial_state(n_qubits);
    State scaluq_state(n_qubits);
    scaluq_state.load(std::vector<scaluq::StdComplex>(initial.begin(), initial.end()));
    auto qulacs_state = make_qulacs_state_parallel(initial);
    const auto scaluq_matrix = make_scaluq_matrix<Prec>(qulacs_matrix);
    const std::uint64_t target_mask = (1ULL << path.target0) | (1ULL << path.target1);

    scaluq::internal::two_target_dense_matrix_gate(
        target_mask, 0, 0, scaluq_matrix, scaluq_state);
    Kokkos::fence();
    double_qubit_dense_matrix_gate_c(path.target0,
                                     path.target1,
                                     qulacs_matrix.data(),
                                     qulacs_state.data(),
                                     qulacs_state.size());
    const auto scaluq_values = scaluq_state.get_amplitudes();
    double max_error = 0.0;
    for (std::size_t i = 0; i < initial.size(); ++i) {
        max_error = std::max(max_error, std::abs(scaluq_values[i] - qulacs_state[i]));
    }
    const double tolerance = Prec == scaluq::Precision::F32 ? 2e-6 : 1e-12;
    if (max_error > tolerance) {
        throw std::runtime_error("state mismatch in " + std::string(path.name));
    }
    return max_error;
}

template <scaluq::Precision Prec>
double dispatch_scaluq(const Options& options,
                       const PathCase& path,
                       std::uint64_t n_qubits,
                       const std::vector<CTYPE>& matrix) {
    if (n_qubits < options.parallel_nqubit_threshold) {
        return measure_scaluq_case<Prec, scaluq::ExecutionSpace::HostSerial>(
            options, path, n_qubits, matrix);
    }
    return measure_scaluq_case<Prec, scaluq::ExecutionSpace::Default>(
        options, path, n_qubits, matrix);
}

template <scaluq::Precision Prec>
double dispatch_validation(const Options& options,
                           const PathCase& path,
                           std::uint64_t n_qubits,
                           const std::vector<CTYPE>& matrix) {
    if (n_qubits < options.parallel_nqubit_threshold) {
        return validate_case<Prec, scaluq::ExecutionSpace::HostSerial>(
            path, n_qubits, matrix);
    }
    return validate_case<Prec, scaluq::ExecutionSpace::Default>(path, n_qubits, matrix);
}

template <class Function>
void sweep_qubits(const Options& options, bool ascending, Function&& function) {
    if (ascending) {
        for (std::uint64_t n = options.min_qubits; n <= options.max_qubits; ++n) {
            function(n, static_cast<std::size_t>(n - options.min_qubits));
        }
        return;
    }
    for (std::uint64_t n = options.max_qubits;; --n) {
        function(n, static_cast<std::size_t>(n - options.min_qubits));
        if (n == options.min_qubits) break;
    }
}

template <scaluq::Precision Prec>
void benchmark_path(std::ofstream& csv,
                    const Options& options,
                    const PathCase& path,
                    const std::vector<CTYPE>& matrix) {
    const char* precision = Prec == scaluq::Precision::F32 ? "f32" : "f64";
    const std::size_t case_count =
        static_cast<std::size_t>(options.max_qubits - options.min_qubits + 1);
    std::vector<CaseResult> results(case_count);

    std::cout << "\n[validate " << precision << ' ' << path.name << "]\n";
    sweep_qubits(options, true, [&](std::uint64_t n, std::size_t index) {
        results[index].max_error = dispatch_validation<Prec>(options, path, n, matrix);
    });

    std::cout << "[Scaluq ascending " << precision << ' ' << path.name << "]\n";
    sweep_qubits(options, true, [&](std::uint64_t n, std::size_t index) {
        results[index].scaluq_ascending_us = dispatch_scaluq<Prec>(options, path, n, matrix);
        std::cout << "q=" << std::setw(2) << n
                  << " median_us=" << results[index].scaluq_ascending_us << '\n';
    });

    std::cout << "[Qulacs ascending f64 " << path.name << "]\n";
    sweep_qubits(options, true, [&](std::uint64_t n, std::size_t index) {
        results[index].qulacs_ascending_us =
            measure_qulacs_case(options, path, n, matrix);
        std::cout << "q=" << std::setw(2) << n
                  << " median_us=" << results[index].qulacs_ascending_us << '\n';
    });

    std::cout << "[Qulacs descending f64 " << path.name << "]\n";
    sweep_qubits(options, false, [&](std::uint64_t n, std::size_t index) {
        results[index].qulacs_descending_us =
            measure_qulacs_case(options, path, n, matrix);
        std::cout << "q=" << std::setw(2) << n
                  << " median_us=" << results[index].qulacs_descending_us << '\n';
    });

    std::cout << "[Scaluq descending " << precision << ' ' << path.name << "]\n";
    sweep_qubits(options, false, [&](std::uint64_t n, std::size_t index) {
        results[index].scaluq_descending_us = dispatch_scaluq<Prec>(options, path, n, matrix);
        std::cout << "q=" << std::setw(2) << n
                  << " median_us=" << results[index].scaluq_descending_us << '\n';
    });

    std::cout << "[balanced " << precision << ' ' << path.name << "]\n";
    sweep_qubits(options, true, [&](std::uint64_t n, std::size_t index) {
        const auto& result = results[index];
        const double scaluq_median =
            median({result.scaluq_ascending_us, result.scaluq_descending_us});
        const double qulacs_median =
            median({result.qulacs_ascending_us, result.qulacs_descending_us});
        csv << n << ',' << path.name << ',' << precision << ",\"{" << path.target0 << ';'
            << path.target1 << "}\"," << scaluq_median << ',' << qulacs_median << ','
            << qulacs_median / scaluq_median << ',' << result.max_error << '\n';
        std::cout << "q=" << std::setw(2) << n << " Scaluq=" << std::setw(11)
                  << scaluq_median << " us Qulacs=" << std::setw(11) << qulacs_median
                  << " us speedup=" << qulacs_median / scaluq_median << "x\n";
    });
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
    std::cout << "parallel_nqubit_threshold=" << options.parallel_nqubit_threshold << '\n';

    std::cout << "\n[f32 Scaluq versus f64 Qulacs]\n";
    for (const auto& path : path_cases) {
        benchmark_path<scaluq::Precision::F32>(csv, options, path, matrix);
    }
    std::cout << "\n[f64 Scaluq versus f64 Qulacs]\n";
    for (const auto& path : path_cases) {
        if (!path.supports_f64) continue;
        benchmark_path<scaluq::Precision::F64>(csv, options, path, matrix);
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
