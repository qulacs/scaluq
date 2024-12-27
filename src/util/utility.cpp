#include <scaluq/util/utility.hpp>

#include "template.hpp"

namespace scaluq {
namespace internal {
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
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
CALL_MACRO_FOR_COMPLEX(FUNC_MACRO)
CALL_MACRO_FOR_UINT(FUNC_MACRO)
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
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
CALL_MACRO_FOR_COMPLEX(FUNC_MACRO)
CALL_MACRO_FOR_UINT(FUNC_MACRO)
#undef FUNC_MACRO

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
CALL_MACRO_FOR_COMPLEX(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(T)                                                           \
    template std::vector<std::vector<T>> convert_2d_device_view_to_host_vector( \
        const Kokkos::View<T**, Kokkos::LayoutRight>&);
CALL_MACRO_FOR_COMPLEX(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT(Fp)
Matrix<Fp> convert_external_matrix_to_internal_matrix(const ComplexMatrix<Fp>& eigen_matrix) {
    std::uint64_t rows = eigen_matrix.rows();
    std::uint64_t cols = eigen_matrix.cols();
    Kokkos::View<const Complex<Fp>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_view(reinterpret_cast<const Complex<Fp>*>(eigen_matrix.data()), rows, cols);
    Matrix<Fp> mat("internal_matrix", rows, cols);
    Kokkos::deep_copy(mat, host_view);
    return mat;
}
#define FUNC_MACRO(Fp) \
    template Matrix<Fp> convert_external_matrix_to_internal_matrix(const ComplexMatrix<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT(Fp)
ComplexMatrix<Fp> convert_internal_matrix_to_external_matrix(const Matrix<Fp>& matrix) {
    int rows = matrix.extent(0);
    int cols = matrix.extent(1);
    ComplexMatrix<Fp> eigen_matrix(rows, cols);
    Kokkos::View<Complex<Fp>**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_view(reinterpret_cast<Complex<Fp>*>(eigen_matrix.data()), rows, cols);
    Kokkos::deep_copy(host_view, matrix);
    return eigen_matrix;
}
#define FUNC_MACRO(Fp) \
    template ComplexMatrix<Fp> convert_internal_matrix_to_external_matrix(const Matrix<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT(Fp)
ComplexMatrix<Fp> convert_coo_to_external_matrix(SparseMatrix<Fp> mat) {
    ComplexMatrix<Fp> eigen_matrix = ComplexMatrix<Fp>::Zero(mat._row, mat._col);
    auto vec_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mat._values);
    for (std::size_t i = 0; i < mat._values.extent(0); i++) {
        eigen_matrix(vec_h(i).r, vec_h(i).c) = vec_h(i).val;
    }
    return eigen_matrix;
}
#define FUNC_MACRO(Fp) template ComplexMatrix<Fp> convert_coo_to_external_matrix(SparseMatrix<Fp>);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

}  // namespace internal
}  // namespace scaluq
