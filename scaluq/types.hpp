#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

namespace scaluq {

using InitializationSettings = Kokkos::InitializationSettings;

inline void initialize(const InitializationSettings& settings = InitializationSettings()) {
    Kokkos::initialize(settings);
}
inline void finalize() { Kokkos::finalize(); }

using UINT = std::uint64_t;

using Complex = Kokkos::complex<double>;
using namespace std::complex_literals;

using StdComplex = std::complex<double>;
using ComplexMatrix = Eigen::Matrix<StdComplex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseComplexMatrix = Eigen::SparseMatrix<StdComplex>;

using device_type = typename Kokkos::Device<Kokkos::DefaultExecutionSpace,
                                            typename Kokkos::DefaultExecutionSpace::memory_space>;
using CrsMatrix = typename KokkosSparse::
    CrsMatrix<default_scalar, default_lno_t, device_type, void, default_size_type>;

struct array_4 {
    Complex val[4];
};

struct matrix_2_2 {
    Complex val[2][2];
};

struct matrix_4_4 {
    Complex val[4][4];
};

struct diagonal_matrix_2_2 {
    Complex val[2];
};

}  // namespace scaluq
