#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <any>
#include <complex>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "type/complex.hpp"
#include "type/floating_point.hpp"

namespace scaluq {
using StdComplex = std::complex<double>;
using Json = nlohmann::json;

#ifdef SCALUQ_USE_CUDA
enum class ExecutionSpace { Host, HostSerial, Default };
#else
enum class ExecutionSpace { Host, HostSerial, Default = Host };
#endif

using ComplexMatrix = Eigen::Matrix<StdComplex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseComplexMatrix = Eigen::SparseMatrix<StdComplex, Eigen::RowMajor>;

namespace internal {
template <typename SpaceType>
concept KokkosExecutionSpace = Kokkos::is_execution_space_v<SpaceType>;

template <ExecutionSpace Space>
struct SpaceTypeImpl {};
template <>
struct SpaceTypeImpl<ExecutionSpace::Host> {
    using Type = Kokkos::DefaultHostExecutionSpace;
};
template <>
struct SpaceTypeImpl<ExecutionSpace::HostSerial> {
    using Type = Kokkos::Serial;
};
#ifdef SCALUQ_USE_CUDA
template <>
struct SpaceTypeImpl<ExecutionSpace::Default> {
    using Type = Kokkos::DefaultExecutionSpace;
};
#endif
template <ExecutionSpace Space>
using SpaceType = typename SpaceTypeImpl<Space>::Type;

template <typename DummyType>
constexpr bool lazy_false_v = false;  // Used for lazy evaluation in static_assert.

template <Precision Prec, ExecutionSpace Space>
using Matrix = Kokkos::View<Complex<Prec>**, Kokkos::LayoutRight, SpaceType<Space>>;

template <Precision Prec>
using Matrix2x2 = Kokkos::Array<Kokkos::Array<Complex<Prec>, 2>, 2>;
template <Precision Prec>
using Matrix4x4 = Kokkos::Array<Kokkos::Array<Complex<Prec>, 4>, 4>;
template <Precision Prec>
using DiagonalMatrix2x2 = Kokkos::Array<Complex<Prec>, 2>;

template <Precision Prec, ExecutionSpace Space>
class SparseMatrix {
    using FloatType = internal::Float<Prec>;
    using ComplexType = internal::Complex<Prec>;

public:
    Kokkos::View<ComplexType*, SpaceType<Space>> _vals;
    Kokkos::View<std::uint32_t*, SpaceType<Space>> _col_idx, _row_ptr;
    std::uint64_t _rows, _cols;

    SparseMatrix(const SparseComplexMatrix& sp);
};
}  // namespace internal

class ConcurrentStream {
public:
    using ExecutionSpaceType = internal::SpaceType<ExecutionSpace::Default>;
    using FenceFn = void (*)(const std::any&, const std::string&);

    ConcurrentStream() : ConcurrentStream(ExecutionSpaceType{}) {}

    template <typename SpaceType>
    requires internal::KokkosExecutionSpace<SpaceType>
    explicit ConcurrentStream(const SpaceType& space)
        : _space(space), _fence(&ConcurrentStream::fence_impl<SpaceType>) {}

    void fence(const std::string& name = "scaluq::ConcurrentStream::fence") const {
        _fence(_space, name);
    }

    [[nodiscard]] const ExecutionSpaceType& get() const { return get<ExecutionSpaceType>(); }

    template <typename SpaceType>
    requires internal::KokkosExecutionSpace<SpaceType>
    [[nodiscard]] const SpaceType& get() const {
        const auto* space = std::any_cast<SpaceType>(&_space);
        if (space == nullptr) {
            throw std::runtime_error(
                "Error: ConcurrentStream::get requested execution space does not match the "
                "contained space type.");
        }
        return *space;
    }

private:
    template <typename SpaceType>
    requires internal::KokkosExecutionSpace<SpaceType>
    static void fence_impl(const std::any& any_space, const std::string& name) {
        std::any_cast<const SpaceType&>(any_space).fence(name);
    }

    std::any _space;
    FenceFn _fence = nullptr;
};

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

template <>
struct adl_serializer<::scaluq::ComplexMatrix> {
    static void to_json(json& j, const ::scaluq::ComplexMatrix& value) {
        j = json::array();
        for (int row_idx = 0; row_idx < value.rows(); ++row_idx) {
            json row = json::array();
            for (int col_idx = 0; col_idx < value.cols(); ++col_idx) {
                row.push_back(value(row_idx, col_idx));
            }
            j.push_back(row);
        }
    }
    static void from_json(const json& j, ::scaluq::ComplexMatrix& value) {
        int rows = j.size();
        int cols = j[0].size();
        value.resize(rows, cols);
        int row_idx = 0;
        for (const auto& row : j) {
            int col_idx = 0;
            for (const auto& val : row) {
                value(row_idx, col_idx) = val.get<::scaluq::StdComplex>();
                ++col_idx;
            }
            ++row_idx;
        }
    }
};

template <>
struct adl_serializer<::scaluq::SparseComplexMatrix> {
    static void to_json(json& j, const ::scaluq::SparseComplexMatrix& value) {
        j["rows"] = value.rows();
        j["cols"] = value.cols();
        json triplets = json::array();
        for (std::uint64_t row_idx = 0; row_idx < static_cast<std::uint64_t>(value.outerSize());
             ++row_idx) {
            for (typename ::scaluq::SparseComplexMatrix::InnerIterator it(value, row_idx); it;
                 ++it) {
                triplets.push_back({{"row", it.row()}, {"col", it.col()}, {"val", it.value()}});
            }
        }
        j["triplets"] = triplets;
    }
    static void from_json(const json& j, ::scaluq::SparseComplexMatrix& value) {
        std::uint64_t rows = j["rows"].get<std::uint64_t>();
        std::uint64_t cols = j["cols"].get<std::uint64_t>();
        value.resize(rows, cols);
        std::vector<Eigen::Triplet<::scaluq::StdComplex>> triplets;
        for (const auto& triplet : j["triplets"]) {
            triplets.emplace_back(triplet["row"].get<std::uint64_t>(),
                                  triplet["col"].get<std::uint64_t>(),
                                  triplet["val"].get<::scaluq::StdComplex>());
        }
        value.setFromTriplets(triplets.begin(), triplets.end());
    }
};
}  // namespace nlohmann

#ifdef SCALUQ_USE_NANOBIND
#include <nanobind/nanobind.h>
namespace scaluq::internal {
namespace nb = nanobind;
}
#endif
