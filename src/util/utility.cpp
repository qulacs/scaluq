#include <scaluq/util/utility.hpp>

#include "../prec_space.hpp"

namespace scaluq {
namespace internal {
template <>
Matrix<Prec, Space> convert_external_matrix_to_internal_matrix<Prec, Space>(
    const ComplexMatrix& eigen_matrix) {
    std::uint64_t rows = eigen_matrix.rows();
    std::uint64_t cols = eigen_matrix.cols();
    if constexpr (Prec == Precision::F64) {
        Matrix<Prec, Space> mat_f64("internal_matrix", rows, cols);
        Kokkos::
            View<const Complex<Prec>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                host_view(reinterpret_cast<const Complex<Prec>*>(eigen_matrix.data()), rows, cols);
        Kokkos::deep_copy(mat_f64, host_view);
        return mat_f64;
    } else {
        Kokkos::View<StdComplex**, Kokkos::LayoutRight, SpaceType<Space>> mat_f64(
            "internal_matrix", rows, cols);
        Kokkos::View<const StdComplex**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            host_view(reinterpret_cast<const StdComplex*>(eigen_matrix.data()), rows, cols);
        Kokkos::deep_copy(mat_f64, host_view);
        Matrix<Prec, Space> mat("internal_matrix", rows, cols);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, SpaceType<Space>>({0, 0}, {rows, cols}),
            KOKKOS_LAMBDA(std::uint64_t r, std::uint64_t c) {
                mat(r, c) = Complex<Prec>(static_cast<Float<Prec>>(mat_f64(r, c).real()),
                                          static_cast<Float<Prec>>(mat_f64(r, c).imag()));
            });
        return mat;
    }
}

template <>
ComplexMatrix convert_internal_matrix_to_external_matrix<Prec, Space>(
    const Matrix<Prec, Space>& matrix) {
    int rows = matrix.extent(0);
    int cols = matrix.extent(1);
    ComplexMatrix eigen_matrix(rows, cols);
    if constexpr (Prec == Precision::F64) {
        Kokkos::View<Complex<Prec>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            host_view(reinterpret_cast<Complex<Prec>*>(eigen_matrix.data()), rows, cols);
        Kokkos::deep_copy(host_view, matrix);
    } else {
        Kokkos::View<StdComplex**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            host_view(reinterpret_cast<StdComplex*>(eigen_matrix.data()), rows, cols);
        Matrix<Prec, Space> matrix_f64("internal_matrix", rows, cols);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, SpaceType<Space>>({0, 0}, {rows, cols}),
            KOKKOS_LAMBDA(std::uint64_t r, std::uint64_t c) {
                matrix_f64(r, c) = StdComplex(static_cast<double>(matrix(r, c).real()),
                                              static_cast<double>(matrix(r, c).imag()));
            });
        Kokkos::deep_copy(host_view, matrix_f64);
    }
    return eigen_matrix;
}

template <>
ComplexMatrix convert_csr_to_external_matrix<Prec, Space>(SparseMatrix<Prec, Space> mat) {
    ComplexMatrix eigen_matrix = ComplexMatrix::Zero(mat._rows, mat._cols);
    auto vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat._vals);
    auto col_idx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat._col_idx);
    auto row_ptr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat._row_ptr);
    for (std::uint32_t i = 0; i < mat._rows; ++i) {
        for (std::uint32_t idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            eigen_matrix(i, col_idx[idx]) = vals[idx];
        }
    }
    return eigen_matrix;
}

}  // namespace internal
}  // namespace scaluq
