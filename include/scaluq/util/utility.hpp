#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <ranges>
#include <vector>

// #include "../operator/pauli_operator.hpp"
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

template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> matrix_multiply(const Matrix2x2<Fp>& matrix1,
                                                     const Matrix2x2<Fp>& matrix2) {
    return {matrix1[0][0] * matrix2[0][0] + matrix1[0][1] * matrix2[1][0],
            matrix1[0][0] * matrix2[0][1] + matrix1[0][1] * matrix2[1][1],
            matrix1[1][0] * matrix2[0][0] + matrix1[1][1] * matrix2[1][0],
            matrix1[1][0] * matrix2[0][1] + matrix1[1][1] * matrix2[1][1]};
}

template <std::floating_point Fp>
inline internal::ComplexMatrix<Fp> kronecker_product(const internal::ComplexMatrix<Fp>& lhs,
                                                     const internal::ComplexMatrix<Fp>& rhs) {
    internal::ComplexMatrix<Fp> result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int j = 0; j < lhs.cols(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
}

template <std::floating_point Fp>
inline internal::ComplexMatrix<Fp> get_expanded_matrix(
    const internal::ComplexMatrix<Fp>& from_matrix,
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
    internal::ComplexMatrix<Fp> to_matrix =
        internal::ComplexMatrix<Fp>::Zero(1ULL << to_targets.size(), 1ULL << to_targets.size());
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
Kokkos::View<T*> convert_host_vector_to_device_view(const std::vector<T>& vec);

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T>
std::vector<T> convert_device_view_to_host_vector(const Kokkos::View<T*>& device_view);

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T, typename Layout>
std::vector<std::vector<T>> convert_2d_device_view_to_host_vector(
    const Kokkos::View<T**, Layout>& view_d);

template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION double squared_norm(const Complex<Fp>& z) {
    return z.real() * z.real() + z.imag() * z.imag();
}

template <std::floating_point Fp>
Matrix<Fp> convert_external_matrix_to_internal_matrix(const ComplexMatrix<Fp>& eigen_matrix);

template <std::floating_point Fp>
ComplexMatrix<Fp> convert_internal_matrix_to_external_matrix(const Matrix<Fp>& matrix);

template <std::floating_point Fp>
ComplexMatrix<Fp> convert_coo_to_external_matrix(SparseMatrix<Fp> mat);

template <std::floating_point Fp>
inline ComplexMatrix<Fp> transform_dense_matrix_by_order(
    const ComplexMatrix<Fp>& mat, const std::vector<std::uint64_t>& targets) {
    std::vector<std::uint64_t> sorted(targets);
    std::sort(sorted.begin(), sorted.end());

    const std::size_t matrix_size = mat.rows();
    const std::uint64_t n_targets = targets.size();

    std::vector<std::uint64_t> targets_order(n_targets);
    for (std::size_t i = 0; i < n_targets; i++) {
        targets_order[i] =
            std::lower_bound(sorted.begin(), sorted.end(), targets[i]) - sorted.begin();
    }

    // transform_indices
    std::vector<std::uint64_t> transformed(matrix_size);
    for (std::size_t index = 0; index < matrix_size; index++) {
        for (std::size_t j = 0; j < targets_order.size(); j++) {
            transformed[index] |= ((index >> targets_order[j]) & 1ULL) << j;
        }
    }

    ComplexMatrix<Fp> ret(matrix_size, matrix_size);

    for (std::size_t i = 0; i < matrix_size; i++) {
        std::size_t row_src = transformed[i];
        for (std::size_t j = 0; j < matrix_size; j++) {
            std::size_t col_src = transformed[j];
            ret(i, j) = mat(row_src, col_src);
        }
    }
    return ret;
}

template <std::floating_point Fp>
inline SparseComplexMatrix<Fp> transform_sparse_matrix_by_order(
    // This is temporary implementation.
    // SparseComplexMatrix will be replaced with std::vector<std::vector<std::Complex<double>>>.
    const SparseComplexMatrix<Fp>& mat,
    const std::vector<std::uint64_t>& targets) {
    return transform_dense_matrix_by_order<Fp>(mat.toDense(), targets).sparseView();
}

}  // namespace internal

}  // namespace scaluq
