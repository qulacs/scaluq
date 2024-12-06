#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "kokkos.hpp"

namespace scaluq {
template <std::floating_point Fp>
using StdComplex = std::complex<Fp>;
template <std::floating_point Fp>
using Complex = Kokkos::complex<Fp>;
using Json = nlohmann::json;

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

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_types_hpp(nb::module_& m) {
    m.def("finalize",
          &finalize,
          "Terminate the Kokkos execution environment. Release the resources.\n\n.. note:: "
          "Finalization fails if there exists `StateVector` allocated. You must use "
          "`StateVector` only inside inner scopes than the usage of `finalize` or delete all of "
          "existing `StateVector`.\n\n.. note:: This is "
          "automatically called when the program exits. If you call this manually, you cannot use "
          "most of scaluq's functions until the program exits.");
    m.def("is_finalized", &is_initialized, "Return true if `finalize()` is already called.");
}
}  // namespace internal
#endif

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
