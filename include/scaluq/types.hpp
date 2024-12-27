#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "kokkos.hpp"
#include "type/complex.hpp"
#include "type/floating_point.hpp"

namespace scaluq {
template <FloatingPoint Fp>
using StdComplex = std::complex<Fp>;
template <FloatingPoint Fp>
using Complex = Kokkos::complex<Fp>;
using Json = nlohmann::json;

namespace internal {
template <typename DummyType>
constexpr bool lazy_false_v = false;  // Used for lazy evaluation in static_assert.

template <FloatingPoint Fp>
using ComplexMatrix =
    Eigen::Matrix<StdComplex<Fp>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <FloatingPoint Fp>
using SparseComplexMatrix = Eigen::SparseMatrix<StdComplex<Fp>, Eigen::RowMajor>;

template <FloatingPoint Fp>
using Matrix = Kokkos::View<Complex<Fp>**, Kokkos::LayoutRight>;

template <FloatingPoint Fp>
using Matrix2x2 = Kokkos::Array<Kokkos::Array<Complex<Fp>, 2>, 2>;
template <FloatingPoint Fp>
using Matrix4x4 = Kokkos::Array<Kokkos::Array<Complex<Fp>, 4>, 4>;
template <FloatingPoint Fp>
using DiagonalMatrix2x2 = Kokkos::Array<Complex<Fp>, 2>;
template <FloatingPoint Fp>
struct SparseValue {
    Complex<Fp> val;
    uint32_t r, c;
};

template <FloatingPoint Fp>
class SparseMatrix {
public:
    Kokkos::View<SparseValue<Fp>*> _values;
    std::uint64_t _row, _col;

    SparseMatrix(const SparseComplexMatrix<Fp>& sp);
};
}  // namespace internal
}  // namespace scaluq

namespace nlohmann {
template <std::floating_point Fp>
struct adl_serializer<scaluq::Complex<Fp>> {
    static void to_json(json& j, const scaluq::Complex<Fp>& c) {
        j = json{{"real", c.real()}, {"imag", c.imag()}};
    }
    static void from_json(const json& j, scaluq::Complex<Fp>& c) {
        j.at("real").get_to(c.real());
        j.at("imag").get_to(c.imag());
    }
};
}  // namespace nlohmann
