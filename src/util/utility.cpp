#include <scaluq/util/utility.hpp>

#include "template.hpp"

namespace scaluq {
namespace internal {
<<<<<<< HEAD
// Host std::vector を Device Kokkos::View に変換する関数
template <typename T>
Kokkos::View<T*> convert_host_vector_to_device_view(const std::vector<T>& vec) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        vec.data(), vec.size());
    Kokkos::View<T*> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}
#define FUNC_MACRO(T) \
    template Kokkos::View<T*> convert_host_vector_to_device_view(const std::vector<T>&);
SCALUQ_CALL_MACRO_FOR_TYPES(FUNC_MACRO)
#undef FUNC_MACRO

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T>
std::vector<T> convert_device_view_to_host_vector(const Kokkos::View<T*>& device_view) {
    std::vector<T> host_vector(device_view.extent(0));
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        host_vector.data(), host_vector.size());
    Kokkos::deep_copy(host_view, device_view);
    return host_vector;
}
#define FUNC_MACRO(T) \
    template std::vector<T> convert_device_view_to_host_vector(const Kokkos::View<T*>&);
SCALUQ_CALL_MACRO_FOR_TYPES(FUNC_MACRO)
#undef FUNC_MACRO
=======
>>>>>>> set-space

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T, typename Layout>
std::vector<std::vector<T>> convert_2d_device_view_to_host_vector(
    const Kokkos::View<T**, Layout>& view_d) {
    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_d);
    std::vector<std::vector<T>> result(view_d.extent(0), std::vector<T>(view_d.extent(1), T{0}));
    for (std::size_t i = 0; i < view_d.extent(0); ++i) {
        for (std::size_t j = 0; j < view_d.extent(1); ++j) {
            result[i][j] = view_h(i, j);
        }
    }
    return result;
}
#define FUNC_MACRO(T)                                                           \
    template std::vector<std::vector<T>> convert_2d_device_view_to_host_vector( \
        const Kokkos::View<T**, Kokkos::LayoutLeft>&);
SCALUQ_CALL_MACRO_FOR_TYPES(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(T)                                                           \
    template std::vector<std::vector<T>> convert_2d_device_view_to_host_vector( \
        const Kokkos::View<T**, Kokkos::LayoutRight>&);
SCALUQ_CALL_MACRO_FOR_TYPES(FUNC_MACRO)
#undef FUNC_MACRO

<<<<<<< HEAD
template <Precision Prec>
Matrix<Prec> convert_external_matrix_to_internal_matrix(const ComplexMatrix& eigen_matrix) {
    std::uint64_t rows = eigen_matrix.rows();
    std::uint64_t cols = eigen_matrix.cols();
    Matrix<Precision::F64> mat_f64("internal_matrix", rows, cols);
    Kokkos::View<const Complex<Prec>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_view(reinterpret_cast<const Complex<Prec>*>(eigen_matrix.data()), rows, cols);
    Kokkos::deep_copy(mat_f64, host_view);
    if constexpr (Prec == Precision::F64) return mat_f64;
    Matrix<Prec> mat("internal_matrix", rows, cols);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {rows, cols}),
        KOKKOS_LAMBDA(std::uint64_t r, std::uint64_t c) {
            mat(r, c) = Complex<Prec>(static_cast<Float<Prec>>(mat_f64(r, c).real()),
                                      static_cast<Float<Prec>>(mat_f64(r, c).imag()));
        });
    return mat;
}
#define FUNC_MACRO(Prec) \
    template Matrix<Prec> convert_external_matrix_to_internal_matrix(const ComplexMatrix&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
ComplexMatrix convert_internal_matrix_to_external_matrix(const Matrix<Prec>& matrix) {
    int rows = matrix.extent(0);
    int cols = matrix.extent(1);
    ComplexMatrix eigen_matrix(rows, cols);
    Kokkos::
        View<Complex<Precision::F64>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            host_view(reinterpret_cast<Complex<Precision::F64>*>(eigen_matrix.data()), rows, cols);
    if constexpr (Prec == Precision::F64) {
        Kokkos::deep_copy(host_view, matrix);
    } else {
        Matrix<Precision::F64> matrix_f64("internal_matrix", rows, cols);
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {rows, cols}),
            KOKKOS_LAMBDA(std::uint64_t r, std::uint64_t c) {
                matrix_f64(r, c) = Complex<Precision::F64>(
                    static_cast<Float<Precision::F64>>(matrix(r, c).real()),
                    static_cast<Float<Precision::F64>>(matrix(r, c).imag()));
            });
        Kokkos::deep_copy(host_view, matrix_f64);
    }
    return eigen_matrix;
}
#define FUNC_MACRO(Prec) \
    template ComplexMatrix convert_internal_matrix_to_external_matrix(const Matrix<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
ComplexMatrix convert_coo_to_external_matrix(SparseMatrix<Prec> mat) {
    ComplexMatrix eigen_matrix = ComplexMatrix::Zero(mat._row, mat._col);
    auto vec_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat._values);
    for (std::size_t i = 0; i < mat._values.extent(0); i++) {
        eigen_matrix(vec_h(i).r, vec_h(i).c) = static_cast<StdComplex>(vec_h(i).val);
    }
    return eigen_matrix;
}
#define FUNC_MACRO(Prec) template ComplexMatrix convert_coo_to_external_matrix(SparseMatrix<Prec>);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
=======
template <std::floating_point Fp, ExecutionSpace Sp>
Matrix<Fp, Sp> convert_external_matrix_to_internal_matrix(const ComplexMatrix<Fp>& eigen_matrix) {
    std::uint64_t rows = eigen_matrix.rows();
    std::uint64_t cols = eigen_matrix.cols();
    Kokkos::View<const Complex<Fp>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_view(reinterpret_cast<const Complex<Fp>*>(eigen_matrix.data()), rows, cols);
    Matrix<Fp, Sp> mat("internal_matrix", rows, cols);
    Kokkos::deep_copy(mat, host_view);
    return mat;
}
#define FUNC_MACRO(Fp, Sp) \
    template Matrix<Fp, Sp> convert_external_matrix_to_internal_matrix(const ComplexMatrix<Fp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp, ExecutionSpace Sp>
ComplexMatrix<Fp> convert_internal_matrix_to_external_matrix(const Matrix<Fp, Sp>& matrix) {
    int rows = matrix.extent(0);
    int cols = matrix.extent(1);
    ComplexMatrix<Fp> eigen_matrix(rows, cols);
    Kokkos::View<Complex<Fp>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_view(reinterpret_cast<Complex<Fp>*>(eigen_matrix.data()), rows, cols);
    Kokkos::deep_copy(host_view, matrix);
    return eigen_matrix;
}
#define FUNC_MACRO(Fp, Sp) \
    template ComplexMatrix<Fp> convert_internal_matrix_to_external_matrix(const Matrix<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp, ExecutionSpace Sp>
ComplexMatrix<Fp> convert_coo_to_external_matrix(const SparseMatrix<Fp, Sp>& mat) {
    ComplexMatrix<Fp> eigen_matrix = ComplexMatrix<Fp>::Zero(mat._row, mat._col);
    auto vec_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat._values);
    for (std::size_t i = 0; i < mat._values.extent(0); i++) {
        eigen_matrix(vec_h(i).r, vec_h(i).c) = vec_h(i).val;
    }
    return eigen_matrix;
}
#define FUNC_MACRO(Fp, Sp) \
    template ComplexMatrix<Fp> convert_coo_to_external_matrix(const SparseMatrix<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
>>>>>>> set-space
#undef FUNC_MACRO

}  // namespace internal
}  // namespace scaluq
