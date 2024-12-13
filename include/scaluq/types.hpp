#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

#include "kokkos.hpp"

namespace scaluq {
template <std::floating_point Fp>
using StdComplex = std::complex<Fp>;
template <std::floating_point Fp>
using Complex = Kokkos::complex<Fp>;

namespace internal {
template <typename DummyType>
constexpr bool lazy_false_v = false;  // Used for lazy evaluation in static_assert.

template <std::floating_point Fp>
using ComplexMatrix =
    Eigen::Matrix<StdComplex<Fp>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <std::floating_point Fp>
using SparseComplexMatrix = Eigen::SparseMatrix<StdComplex<Fp>, Eigen::RowMajor>;

template <std::floating_point Fp>
using Matrix = Kokkos::View<Complex<Fp>**, Kokkos::LayoutRight>;

template <std::floating_point Fp>
using Matrix2x2 = Kokkos::Array<Kokkos::Array<Complex<Fp>, 2>, 2>;
template <std::floating_point Fp>
using Matrix4x4 = Kokkos::Array<Kokkos::Array<Complex<Fp>, 4>, 4>;
template <std::floating_point Fp>
using DiagonalMatrix2x2 = Kokkos::Array<Complex<Fp>, 2>;
template <std::floating_point Fp>
struct SparseValue {
    Complex<Fp> val;
    uint32_t r, c;
};

template <std::floating_point Fp>
class SparseMatrix {
public:
    Kokkos::View<SparseValue<Fp>*> _values;
    std::uint64_t _row, _col;

    SparseMatrix(const SparseComplexMatrix<Fp>& sp);
};
}  // namespace internal
}  // namespace scaluq
