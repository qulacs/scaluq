#include <scaluq/util/utility.hpp>

#include "template.hpp"

namespace scaluq {
namespace internal {
// std::vector を Kokkos::View に変換する関数
template <typename T, ExecutionSpace Space>
inline Kokkos::View<T*, SpaceType<Space>> convert_vector_to_view(const std::vector<T>& vec) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        vec.data(), vec.size());
    Kokkos::View<T*, SpaceType<Space>> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}
#define FUNC_MACRO(T, Space)                                                      \
    template Kokkos::View<T*, SpaceType<Space>> convert_vector_to_view<T, Space>( \
        const std::vector<T>&);
SCALUQ_CALL_MACRO_FOR_TYPES_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

// Kokkos::View を std::vector に変換する関数
template <typename T, ExecutionSpace Space>
inline std::vector<T> convert_view_to_vector(
    const Kokkos::View<T*, SpaceType<Space>>& device_view) {
    std::vector<T> host_vector(device_view.extent(0));
    auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_view);
    std::copy(host_view.data(), host_view.data() + host_view.size(), host_vector.begin());
    return host_vector;
}
#define FUNC_MACRO(T, Space)                                  \
    template std::vector<T> convert_view_to_vector<T, Space>( \
        const Kokkos::View<T*, SpaceType<Space>>&);
SCALUQ_CALL_MACRO_FOR_TYPES_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
inline Matrix<Prec, Space> convert_external_matrix_to_internal_matrix(
    const ComplexMatrix& eigen_matrix) {
    std::uint64_t rows = eigen_matrix.rows();
    std::uint64_t cols = eigen_matrix.cols();
    Matrix<Precision::F64, Space> mat_f64("internal_matrix", rows, cols);
    Kokkos::View<const Complex<Precision::F64>**,
                 Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_view(
            reinterpret_cast<const Complex<Precision::F64>*>(eigen_matrix.data()), rows, cols);
    Kokkos::deep_copy(mat_f64, host_view);
    if constexpr (Prec == Precision::F64) return mat_f64;
    Matrix<Prec, Space> mat("internal_matrix", rows, cols);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, SpaceType<Space>>({0, 0}, {rows, cols}),
        KOKKOS_LAMBDA(std::uint64_t r, std::uint64_t c) {
            mat(r, c) = Complex<Prec>(static_cast<Float<Prec>>(mat_f64(r, c).real()),
                                      static_cast<Float<Prec>>(mat_f64(r, c).imag()));
        });
    return mat;
}
#define FUNC_MACRO(Prec, Space)                                                           \
    template Matrix<Prec, Space> convert_external_matrix_to_internal_matrix<Prec, Space>( \
        const ComplexMatrix&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
inline ComplexMatrix convert_internal_matrix_to_external_matrix(const Matrix<Prec, Space>& matrix) {
    int rows = matrix.extent(0);
    int cols = matrix.extent(1);
    ComplexMatrix eigen_matrix(rows, cols);
    Kokkos::
        View<Complex<Precision::F64>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            host_view(reinterpret_cast<Complex<Precision::F64>*>(eigen_matrix.data()), rows, cols);
    if constexpr (Prec == Precision::F64) {
        Kokkos::deep_copy(host_view, matrix);
    } else {
        Matrix<Precision::F64, Space> matrix_f64("internal_matrix", rows, cols);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, SpaceType<Space>>({0, 0}, {rows, cols}),
            KOKKOS_LAMBDA(std::uint64_t r, std::uint64_t c) {
                matrix_f64(r, c) = Complex<Precision::F64>(
                    static_cast<Float<Precision::F64>>(matrix(r, c).real()),
                    static_cast<Float<Precision::F64>>(matrix(r, c).imag()));
            });
        Kokkos::deep_copy(host_view, matrix_f64);
    }
    return eigen_matrix;
}
#define FUNC_MACRO(Prec, Space)                                                     \
    template ComplexMatrix convert_internal_matrix_to_external_matrix<Prec, Space>( \
        const Matrix<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
inline ComplexMatrix convert_coo_to_external_matrix(SparseMatrix<Prec, Space> mat) {
    ComplexMatrix eigen_matrix = ComplexMatrix::Zero(mat._row, mat._col);
    auto vec_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat._values);
    for (std::size_t i = 0; i < mat._values.extent(0); i++) {
        eigen_matrix(vec_h(i).r, vec_h(i).c) = static_cast<StdComplex>(vec_h(i).val);
    }
    return eigen_matrix;
}
#define FUNC_MACRO(Prec, Space) \
    template ComplexMatrix convert_coo_to_external_matrix<Prec, Space>(SparseMatrix<Prec, Space>);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

}  // namespace internal
}  // namespace scaluq
