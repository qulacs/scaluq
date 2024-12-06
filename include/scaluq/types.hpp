#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>
#ifndef SCALUQ_USE_CUDA
#include <stdfloat>
#endif

#include "kokkos.hpp"

namespace scaluq {
template <std::floating_point Fp>
using StdComplex = std::complex<Fp>;
template <std::floating_point Fp>
using Complex = Kokkos::complex<Fp>;

#ifdef SCALUQ_FLOAT16
#ifdef SCALUQ_USE_CUDA
using F16 = _Float16;
#else
#ifndef __STDCPP_FLOAT16_T__
static_assert(false && "float16 is not supported")
#endif
    using F16 = std::float16_t;
#endif
#endif
#ifdef SCALUQ_FLOAT32
#ifdef SCALUQ_USE_CUDA
using F32 = float;
#else
#ifndef __STDCPP_FLOAT32_T__
static_assert(false && "float32 is not supported")
#endif
    using F32 = std::float32_t;
#endif
#endif
#ifdef SCALUQ_FLOAT64
#ifdef SCALUQ_USE_CUDA
using F64 = double;
#else
#ifndef __STDCPP_FLOAT64_T__
static_assert(false && "float64 is not supported")
#endif
    using F64 = std::float64_t;
#endif
#endif
#ifdef SCALUQ_BFLOAT16
#ifdef SCALUQ_USE_CUDA
using BF16 = __nv_bfloat16;
#else
#ifndef __STDCPP_BFLOAT16_T__
static_assert(false && "bfloat16 is not supported")
#endif
    using BF16 = std::bfloat16_t;
#endif
#endif

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
