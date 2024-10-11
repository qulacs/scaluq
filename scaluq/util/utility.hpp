#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <ranges>
#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../types.hpp"

namespace scaluq {

namespace internal {

/**
 * Inserts a 0 bit at a specified index in basis_index.
 * Example: insert_zero_to_basis_index(0b1001, 1) -> 0b10001.
 *                                                        ^
 */
KOKKOS_INLINE_FUNCTION std::uint64_t insert_zero_to_basis_index(std::uint64_t basis_index,
                                                                std::uint64_t insert_index) {
    std::uint64_t mask = (1ULL << insert_index) - 1;
    std::uint64_t temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
}

/**
 * Inserts multiple 0 bits at specified positions in basis_index.
 * Example: insert_zero_to_basis_index(0b11111, 0x100101) -> 0b11011010.
 *                                                               ^  ^ ^
 */
KOKKOS_INLINE_FUNCTION std::uint64_t insert_zero_at_mask_positions(std::uint64_t basis_index,
                                                                   std::uint64_t insert_mask) {
    for (std::uint64_t bit_mask = insert_mask; bit_mask;
         bit_mask &= (bit_mask - 1)) {  // loop through set bits
        std::uint64_t lower_mask = ~bit_mask & (bit_mask - 1);
        std::uint64_t upper_mask = ~lower_mask;
        basis_index = ((basis_index & upper_mask) << 1) | (basis_index & lower_mask);
    }
    return basis_index;
}

template <bool enable_validate = true>
inline std::uint64_t vector_to_mask(const std::vector<std::uint64_t>& v) {
    std::uint64_t mask = 0;
    for (auto x : v) {
        if constexpr (enable_validate) {
            if (x >= sizeof(std::uint64_t) * 8) [[unlikely]] {
                throw std::runtime_error("The size of the qubit system must be less than 64.");
            }
            if ((mask >> x) & 1) [[unlikely]] {
                throw std::runtime_error("The specified qubit is duplicated.");
            }
        }
        mask |= 1ULL << x;
    }
    return mask;
}

inline std::vector<std::uint64_t> mask_to_vector(std::uint64_t mask) {
    std::vector<std::uint64_t> indices;
    for (std::uint64_t sub_mask = mask; sub_mask; sub_mask &= (sub_mask - 1)) {
        indices.push_back(std::countr_zero(sub_mask));
    }
    return indices;
}

KOKKOS_INLINE_FUNCTION Matrix2x2 matrix_multiply(const Matrix2x2& matrix1,
                                                 const Matrix2x2& matrix2) {
    return {matrix1[0][0] * matrix2[0][0] + matrix1[0][1] * matrix2[1][0],
            matrix1[0][0] * matrix2[0][1] + matrix1[0][1] * matrix2[1][1],
            matrix1[1][0] * matrix2[0][0] + matrix1[1][1] * matrix2[1][0],
            matrix1[1][0] * matrix2[0][1] + matrix1[1][1] * matrix2[1][1]};
}

inline internal::ComplexMatrix kronecker_product(const internal::ComplexMatrix& lhs,
                                                 const internal::ComplexMatrix& rhs) {
    internal::ComplexMatrix result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int j = 0; j < lhs.cols(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
}

inline internal::ComplexMatrix get_expanded_matrix(const internal::ComplexMatrix& from_matrix,
                                                   const std::vector<std::uint64_t>& from_targets,
                                                   std::vector<std::uint64_t>& to_targets) {
    std::vector<std::uint64_t> targets_map(from_targets.size());
    std::ranges::transform(from_targets, targets_map.begin(), [&](std::uint64_t x) {
        return std::ranges::lower_bound(to_targets, x) - to_targets.begin();
    });
    std::vector<std::uint64_t> idx_map(1ULL << from_targets.size());
    for (std::uint64_t i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (std::uint64_t j : std::views::iota(0ULL, from_targets.size())) {
            idx_map[i] |= (i >> j & 1) << targets_map[j];
        }
    }

    std::uint64_t targets_idx_mask = idx_map.back();
    std::vector<std::uint64_t> outer_indices;
    outer_indices.reserve(1ULL << (to_targets.size() - from_targets.size()));
    for (std::uint64_t i : std::views::iota(0ULL, 1ULL << to_targets.size())) {
        if ((i & targets_idx_mask) == 0) outer_indices.push_back(i);
    }
    internal::ComplexMatrix to_matrix =
        internal::ComplexMatrix::Zero(1ULL << to_targets.size(), 1ULL << to_targets.size());
    for (std::uint64_t i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (std::uint64_t j : std::views::iota(0ULL, 1ULL << from_targets.size())) {
            for (std::uint64_t o : outer_indices) {
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
    std::uint64_t rows = eigen_matrix.rows();
    std::uint64_t cols = eigen_matrix.cols();
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

inline ComplexMatrix convert_coo_to_external_matrix(SparseMatrix mat) {
    ComplexMatrix eigen_matrix(mat._row, mat._col);
    for (std::size_t i = 0; i < mat._values.extent(0); i++) {
        eigen_matrix(mat._values(i).r, mat._values(i).c) = mat._values(i).val;
    }
    return eigen_matrix;
}

inline ComplexMatrix transform_dense_matrix_by_order(const ComplexMatrix& mat,
                                                     const std::vector<std::uint64_t>& targets) {
    std::vector<std::uint64_t> sorted(targets);
    std::sort(sorted.begin(), sorted.end());

    const std::size_t matrix_size = mat.rows();

    std::vector<std::uint64_t> targets_order(targets.size());
    for (std::size_t i = 0; i < targets.size(); i++) {
        targets_order[i] =
            std::lower_bound(sorted.begin(), sorted.end(), targets[i]) - sorted.begin();
    }

    // transform_indices
    std::vector<std::uint64_t> transformed(targets_order.size());
    for (std::size_t index = 0; index < targets_order.size(); index++) {
        for (std::size_t j = 0; j < targets_order.size(); j++) {
            transformed[j] |= ((index & (1ULL << targets_order[j])) >> targets_order[j]) << j;
        }
    }

    ComplexMatrix ret(matrix_size, matrix_size);

    for (std::size_t i = 0; i < matrix_size; i++) {
        std::size_t row_dst = transformed[i];
        for (std::size_t j = 0; j < matrix_size; j++) {
            std::size_t col_dst = transformed[j];
            ret(row_dst, col_dst) = mat(i, j);
        }
    }
    return ret;
}

inline SparseComplexMatrix transform_sparse_matrix_by_order(
    // This is temporary implementation.
    // SparseComplexMatrix will be replaced with std::vector<std::vector<std::Complex<double>>>
    // hence this function will be refactored.
    const SparseComplexMatrix& mat,
    const std::vector<std::uint64_t>& targets) {
    ComplexMatrix dense_mat = mat;
    ComplexMatrix transformed = transform_dense_matrix_by_order(dense_mat, targets);
    return transformed.sparseView();
}

}  // namespace internal

}  // namespace scaluq
