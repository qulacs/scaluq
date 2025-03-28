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
using StdComplex = std::complex<double>;
using Json = nlohmann::json;

enum class ExecutionSpace { Host, Default };

namespace internal {
template <ExecutionSpace Space>
struct SpaceTypeImpl {};
template <>
struct SpaceTypeImpl<ExecutionSpace::Host> {
    using Type = Kokkos::DefaultHostExecutionSpace;
};
template <>
struct SpaceTypeImpl<ExecutionSpace::Default> {
    using Type = Kokkos::DefaultExecutionSpace;
};
template <ExecutionSpace Space>
using SpaceType = typename SpaceTypeImpl<Space>::Type;

template <typename DummyType>
constexpr bool lazy_false_v = false;  // Used for lazy evaluation in static_assert.

using ComplexMatrix = Eigen::Matrix<StdComplex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseComplexMatrix = Eigen::SparseMatrix<StdComplex, Eigen::RowMajor>;

template <Precision Prec, ExecutionSpace Space>
using Matrix = Kokkos::View<Complex<Prec>**, Kokkos::LayoutRight, SpaceType<Space>>;

template <Precision Prec>
using Matrix2x2 = Kokkos::Array<Kokkos::Array<Complex<Prec>, 2>, 2>;
template <Precision Prec>
using Matrix4x4 = Kokkos::Array<Kokkos::Array<Complex<Prec>, 4>, 4>;
template <Precision Prec>
using DiagonalMatrix2x2 = Kokkos::Array<Complex<Prec>, 2>;
template <Precision Prec>
struct SparseValue {
    Complex<Prec> val;
    uint32_t r, c;
};

template <Precision Prec, ExecutionSpace Space>
class SparseMatrix {
public:
    Kokkos::View<SparseValue<Prec>*, SpaceType<Space>> _values;
    std::uint64_t _row, _col;

    SparseMatrix(const SparseComplexMatrix& sp);
};
}  // namespace internal
}  // namespace scaluq

namespace nlohmann {
template <>
struct adl_serializer<::scaluq::StdComplex> {
    static void to_json(json& j, const ::scaluq::StdComplex& value) {
        j["real"] = value.real();
        j["imag"] = value.imag();
    }
    static void from_json(const json& j, ::scaluq::StdComplex& value) {
        value.real(j["real"].get<double>());
        value.imag(j["imag"].get<double>());
    }
};
}  // namespace nlohmann

#ifdef SCALUQ_USE_NANOBIND
#include <nanobind/nanobind.h>
namespace scaluq::internal {
namespace nb = nanobind;
}
#endif
