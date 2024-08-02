#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <ranges>
#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../types.hpp"
#include "KokkosSparse_spmv.hpp"

namespace scaluq {

namespace internal {

/**
 * Inserts a 0 bit at a specified index in basis_index.
 * Example: insert_zero_to_basis_index(0b1001, 1) -> 0b10001.
 *                                                        ^
 */
KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index, UINT insert_index) {
    UINT mask = (1ULL << insert_index) - 1;
    UINT temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
}

KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index,
                                                       UINT basis_mask,
                                                       UINT qubit_index) {
    UINT tmp_basis = (basis_index >> qubit_index) << (qubit_index + 1);
    return tmp_basis + basis_index % basis_mask;
}

/**
 * Inserts two 0 bits at specified indexes in basis_index.
 * Example: insert_two_zero_to_basis_index(0b11001, 1, 5) -> 0b1010001.
 *                                                          ^   ^
 */
KOKKOS_INLINE_FUNCTION UINT insert_two_zero_to_basis_index(UINT basis_index,
                                                           UINT insert_index1,
                                                           UINT insert_index2) {
    auto [lidx, uidx] = Kokkos::minmax(insert_index1, insert_index2);
    UINT lmask = (1ULL << lidx) - 1;
    UINT umask = (1ULL << uidx) - 1;
    basis_index = ((basis_index >> lidx) << (lidx + 1)) | (basis_index & lmask);
    return ((basis_index >> uidx) << (uidx + 1)) | (basis_index & umask);
}

KOKKOS_INLINE_FUNCTION matrix_2_2 matrix_multiply(const matrix_2_2& matrix1,
                                                  const matrix_2_2& matrix2) {
    return {matrix1.val[0][0] * matrix2.val[0][0] + matrix1.val[0][1] * matrix2.val[1][0],
            matrix1.val[0][0] * matrix2.val[0][1] + matrix1.val[0][1] * matrix2.val[1][1],
            matrix1.val[1][0] * matrix2.val[0][0] + matrix1.val[1][1] * matrix2.val[1][0],
            matrix1.val[1][0] * matrix2.val[0][1] + matrix1.val[1][1] * matrix2.val[1][1]};
}

inline ComplexMatrix kronecker_product(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
    ComplexMatrix result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int j = 0; j < lhs.cols(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
}

inline ComplexMatrix get_expanded_matrix(const ComplexMatrix& from_matrix,
                                         const std::vector<UINT>& from_targets,
                                         std::vector<UINT>& to_targets) {
    std::vector<UINT> targets_map(from_targets.size());
    std::ranges::transform(from_targets, targets_map.begin(), [&](UINT x) {
        return std::ranges::lower_bound(to_targets, x) - to_targets.begin();
    });
    std::vector<UINT> idx_map(1ULL << from_targets.size());
    for (UINT i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (UINT j : std::views::iota(0ULL, from_targets.size())) {
            idx_map[i] |= (i >> j & 1) << targets_map[j];
        }
    }

    UINT targets_idx_mask = idx_map.back();
    std::vector<UINT> outer_indices;
    outer_indices.reserve(1ULL << (to_targets.size() - from_targets.size()));
    for (UINT i : std::views::iota(0ULL, 1ULL << to_targets.size())) {
        if ((i & targets_idx_mask) == 0) outer_indices.push_back(i);
    }
    ComplexMatrix to_matrix =
        ComplexMatrix::Zero(1ULL << to_targets.size(), 1ULL << to_targets.size());
    for (UINT i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (UINT j : std::views::iota(0ULL, 1ULL << from_targets.size())) {
            for (UINT o : outer_indices) {
                to_matrix(idx_map[i] | o, idx_map[j] | o) = from_matrix(i, j);
            }
        }
    }
    return to_matrix;
}

// Host std::vector を Device Kokkos::View に変換する関数
template <typename T>
inline Kokkos::View<T*> convert_host_vector_to_device_view(const std::vector<T>& vec) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        vec.data(), vec.size());
    Kokkos::View<T*> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T>
inline std::vector<T> convert_device_view_to_host_vector(const Kokkos::View<T*>& device_view) {
    std::vector<T> host_vector(device_view.extent(0));
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        host_vector.data(), host_vector.size());
    Kokkos::deep_copy(host_view, device_view);
    return host_vector;
}

inline std::vector<UINT> create_matrix_mask_list(const std::vector<UINT> qubit_index_list,
                                                 const UINT qubit_index_count) {
    const UINT matrix_dim = 1ULL << qubit_index_count;
    std::vector<UINT> mask_list(matrix_dim, 0);

    for (UINT cursor = 0; cursor < matrix_dim; cursor++) {
        for (UINT bit_cursor = 0; bit_cursor < qubit_index_count; bit_cursor++) {
            if ((cursor >> bit_cursor) & 1) {
                UINT bit_index = qubit_index_list[bit_cursor];
                mask_list[cursor] ^= (1ULL << bit_index);
            }
        }
    }
    return mask_list;
}

inline std::vector<UINT> create_sorted_ui_list(const std::vector<UINT>& list) {
    std::vector<UINT> sorted_list(list);
    std::sort(sorted_list.begin(), sorted_list.end());
    return sorted_list;
}

inline std::vector<UINT> create_sorted_ui_list_value(const std::vector<UINT>& list, UINT value) {
    std::vector<UINT> sorted_list(list);
    sorted_list.emplace_back(value);
    std::sort(sorted_list.begin(), sorted_list.end());
    return sorted_list;
}

inline std::vector<UINT> create_sorted_ui_list_list(const std::vector<UINT>& list1,
                                                    const UINT size1,
                                                    const std::vector<UINT>& list2,
                                                    const UINT size2) {
    std::vector<UINT> new_array(size1 + size2);
    std::copy(list1.begin(), list1.end(), new_array.begin());
    std::copy(list2.begin(), list2.end(), new_array.begin() + size1);
    std::sort(new_array.begin(), new_array.end());
    return new_array;
}

inline void create_shift_mask_list_from_list_and_value_buf(const std::vector<UINT> array,
                                                           UINT target,
                                                           std::vector<UINT> dst_array,
                                                           std::vector<UINT> dst_mask) {
    UINT size = array.size() + 1;
    dst_array.resize(size - 1);
    dst_mask.resize(size);
    std::copy(array.begin(), array.end(), dst_array.begin());
    dst_array.emplace_back(target);
    sort(dst_array.begin(), dst_array.end());
    for (UINT i = 0; i < size; ++i) {
        dst_mask[i] = (1ULL << dst_array[i]) - 1;
    }
}

inline void create_shift_mask_list_from_list_buf(std::vector<UINT> array,
                                                 std::vector<UINT>& dst_array,
                                                 std::vector<UINT>& dst_mask) {
    UINT size = array.size();
    dst_array.resize(size);
    dst_mask.resize(size);
    std::copy(array.begin(), array.end(), dst_array.begin());
    sort(dst_array.begin(), dst_array.end());
    for (UINT i = 0; i < size; ++i) {
        dst_mask[i] = (1ULL << dst_array[i]) - 1;
    }
}

inline UINT create_control_mask(const std::vector<UINT> qubit_index_list,
                                const std::vector<UINT> value_list) {
    UINT mask = 0;
    for (UINT i = 0; i < qubit_index_list.size(); ++i) {
        mask ^= (1ULL << qubit_index_list[i]) * value_list[i];
    }
    return mask;
}

// x: state vector. output will be stored in y
inline void spmv(const CrsMatrix& matrix,
                 const Kokkos::View<Complex*>& x,
                 Kokkos::View<Complex*>& y) {
    KokkosSparse::spmv("N", 1.0, matrix, x, 0.0, y);
}

// x: state vector, output will be stored in y
inline void gemv(const Matrix matrix, const Kokkos::View<Complex*>& x, Kokkos::View<Complex*>& y) {
    KokkosBlas::gemv("N", 1.0, matrix, x, 0.0, y);
}

// x: rearranged state vector, output will be stored in y
inline void gemm_trans(const Matrix matrix, const Kokkos::View<Complex**>& x, Matrix& y) {
    KokkosBlas::gemm("N", "T", 1.0, matrix, x, 0.0, y);
}

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T, typename Layout>
inline std::vector<std::vector<T>> convert_2d_device_view_to_host_vector(
    const Kokkos::View<T**, Layout>& view_d) {
    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_d);
    std::vector<std::vector<T>> result(view_d.extent(0), std::vector<T>(view_d.extent(1), 0));
    for (size_t i = 0; i < view_d.extent(0); ++i) {
        for (size_t j = 0; j < view_d.extent(1); ++j) {
            result[i][j] = view_h(i, j);
        }
    }
    return result;
}

KOKKOS_INLINE_FUNCTION double squared_norm(const Complex& z) {
    return z.real() * z.real() + z.imag() * z.imag();
}

inline Matrix convert_external_matrix_to_internal_matrix(const ComplexMatrix& eigen_matrix) {
    UINT rows = eigen_matrix.rows();
    UINT cols = eigen_matrix.cols();
    Kokkos::View<const Complex**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host_view(reinterpret_cast<const Complex*>(eigen_matrix.data()), rows, cols);
    Matrix mat("internal_matrix", rows, cols);
    Kokkos::deep_copy(mat, host_view);
    return mat;
}

inline ComplexMatrix convert_internal_matrix_to_external_matrix(const Matrix& matrix) {
    int rows = matrix.extent(0);
    int cols = matrix.extent(1);
    ComplexMatrix eigen_matrix(rows, cols);
    Kokkos::View<Complex**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        reinterpret_cast<Complex*>(eigen_matrix.data()), rows, cols);
    Kokkos::deep_copy(host_view, matrix);
    return eigen_matrix;
}

}  // namespace internal
}  // namespace scaluq
