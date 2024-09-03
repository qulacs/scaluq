#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

namespace scaluq {

using InitializationSettings = Kokkos::InitializationSettings;

inline void initialize(const InitializationSettings& settings = InitializationSettings()) {
    Kokkos::initialize(settings);
}
inline void finalize() { Kokkos::finalize(); }
inline bool is_initialized() { return Kokkos::is_initialized(); }
inline bool is_finalized() { return Kokkos::is_finalized(); }

using StdComplex = std::complex<double>;
using Complex = Kokkos::complex<double>;
using namespace std::complex_literals;

namespace internal {

using ComplexMatrix = Eigen::Matrix<StdComplex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseComplexMatrix = Eigen::SparseMatrix<StdComplex>;

using Matrix2x2 = Kokkos::Array<Kokkos::Array<Complex, 2>, 2>;
using Matrix4x4 = Kokkos::Array<Kokkos::Array<Complex, 4>, 4>;
using DiagonalMatrix2x2 = Kokkos::Array<Complex, 2>;

}  // namespace internal
}  // namespace scaluq
