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

// Converts a vector of indices to a bitmask.
template <bool enable_validate = true>
inline std::uint64_t vector_to_mask(const std::vector<std::uint64_t>& indices) {
    std::uint64_t mask = 0;
    for (auto x : indices) {
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

// Converts a vector of indices and values to a bitmask.
inline std::uint64_t vector_to_mask(const std::vector<std::uint64_t>& indices,
                                    const std::vector<std::uint64_t>& values) {
    std::uint64_t mask = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (values[i] == 1) {
            mask |= 1ULL << indices[i];
        } else if (values[i] != 0) {  // 必ず 0 または 1
            throw std::runtime_error("Invalid value in vector_to_mask: " +
                                     std::to_string(values[i]));
        }
    }
    return mask;
}

// 1 が立っているビットの位置を vector に格納する
inline std::vector<std::uint64_t> mask_to_vector(std::uint64_t mask) {
    std::vector<std::uint64_t> indices;
    for (std::uint64_t sub_mask = mask; sub_mask; sub_mask &= (sub_mask - 1)) {
        indices.push_back(std::countr_zero(sub_mask));
    }
    return indices;
}

// mask のビットうち，indices_mask が 1 であるビット位置のビットを vector に格納する
inline std::vector<std::uint64_t> mask_to_vector(std::uint64_t indices_mask, std::uint64_t mask) {
    std::vector<std::uint64_t> values;
    for (std::uint64_t sub_mask = indices_mask; sub_mask; sub_mask &= (sub_mask - 1)) {
        values.push_back((mask >> std::countr_zero(sub_mask)) & 1);
    }
    return values;
}

inline void resize_and_check_control_values(const std::vector<std::uint64_t>& controls,
                                            std::vector<std::uint64_t>& control_values) {
    if (control_values.empty()) {
        control_values.assign(controls.size(), 1);
    }
    if (controls.size() != control_values.size()) {
        throw std::runtime_error("The size of controls and control_values must be the same.");
    }
}

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> matrix_multiply(const Matrix2x2<Prec>& matrix1,
                                                       const Matrix2x2<Prec>& matrix2) {
    return {matrix1[0][0] * matrix2[0][0] + matrix1[0][1] * matrix2[1][0],
            matrix1[0][0] * matrix2[0][1] + matrix1[0][1] * matrix2[1][1],
            matrix1[1][0] * matrix2[0][0] + matrix1[1][1] * matrix2[1][0],
            matrix1[1][0] * matrix2[0][1] + matrix1[1][1] * matrix2[1][1]};
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

ComplexMatrix get_expanded_matrix(const ComplexMatrix& from_matrix,
                                  const std::vector<std::uint64_t>& from_targets,
                                  std::uint64_t from_control_mask,
                                  std::uint64_t from_control_value_mask,
                                  std::vector<std::uint64_t>& to_operands);

// Host std::vector を Device Kokkos::View に変換する関数
template <typename T, ExecutionSpace Space>
Kokkos::View<T*, SpaceType<Space>> convert_vector_to_view(const std::vector<T>& vec) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        vec.data(), vec.size());
    Kokkos::View<T*, SpaceType<Space>> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T, ExecutionSpace Space>
std::vector<T> convert_view_to_vector(const Kokkos::View<T*, SpaceType<Space>>& device_view) {
    std::vector<T> host_vector(device_view.extent(0));
    auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_view);
    std::copy(host_view.data(), host_view.data() + host_view.size(), host_vector.begin());
    return host_vector;
}

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Float<Prec> squared_norm(const Complex<Prec>& z) {
    return z.real() * z.real() + z.imag() * z.imag();
}

template <Precision Prec, ExecutionSpace Space>
Matrix<Prec, Space> convert_external_matrix_to_internal_matrix(const ComplexMatrix& eigen_matrix);

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix convert_internal_matrix_to_external_matrix(const Matrix<Prec, Space>& matrix);

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix convert_csr_to_external_matrix(SparseMatrix<Prec, Space> mat);

inline ComplexMatrix transform_dense_matrix_by_order(const ComplexMatrix& mat,
                                                     const std::vector<std::uint64_t>& targets) {
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

    ComplexMatrix ret(matrix_size, matrix_size);

    for (std::size_t i = 0; i < matrix_size; i++) {
        std::size_t row_src = transformed[i];
        for (std::size_t j = 0; j < matrix_size; j++) {
            std::size_t col_src = transformed[j];
            ret(i, j) = mat(row_src, col_src);
        }
    }
    return ret;
}

inline SparseComplexMatrix transform_sparse_matrix_by_order(
    // This is temporary implementation.
    // SparseComplexMatrix will be replaced with std::vector<std::vector<std::Complex<double>>>.
    const SparseComplexMatrix& mat,
    const std::vector<std::uint64_t>& targets) {
    return transform_dense_matrix_by_order(mat.toDense(), targets).sparseView();
}

}  // namespace internal

}  // namespace scaluq
