#include "update_ops.hpp"

namespace scaluq::internal {

template <>
void zero_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   Complex<Prec> matrix,
                                   DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t dim = dm.dim();
    const Complex<Prec> matrix_conj = conj(matrix);
    const Complex<Prec> norm = matrix * matrix_conj;
    Kokkos::parallel_for(
        "zero_target_dense_matrix_gate_dm",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dim * dim),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t row = g / dim;
            const std::uint64_t col = g % dim;
            const bool row_active = (row & control_mask) == control_value_mask;
            const bool col_active = (col & control_mask) == control_value_mask;
            if (row_active && col_active)
                dm._raw(row, col) *= norm;
            else if (row_active)
                dm._raw(row, col) *= matrix;
            else if (col_active)
                dm._raw(row, col) *= matrix_conj;
        });
}

template <>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t dim = dm.dim();
    const std::uint64_t lower_target_mask = -target_mask & target_mask;
    const std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    const std::uint64_t n_quads = dim >> std::popcount(target_mask | control_mask);

    Kokkos::parallel_for(
        "two_target_dense_matrix_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_quads * dim),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t it = g / dim;
            const std::uint64_t col = g % dim;
            const std::uint64_t row0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            const std::uint64_t row1 = row0 | lower_target_mask;
            const std::uint64_t row2 = row0 | upper_target_mask;
            const std::uint64_t row3 = row0 | target_mask;
            const Complex<Prec> v0 = dm._raw(row0, col);
            const Complex<Prec> v1 = dm._raw(row1, col);
            const Complex<Prec> v2 = dm._raw(row2, col);
            const Complex<Prec> v3 = dm._raw(row3, col);
            dm._raw(row0, col) =
                matrix[0][0] * v0 + matrix[0][1] * v1 + matrix[0][2] * v2 + matrix[0][3] * v3;
            dm._raw(row1, col) =
                matrix[1][0] * v0 + matrix[1][1] * v1 + matrix[1][2] * v2 + matrix[1][3] * v3;
            dm._raw(row2, col) =
                matrix[2][0] * v0 + matrix[2][1] * v1 + matrix[2][2] * v2 + matrix[2][3] * v3;
            dm._raw(row3, col) =
                matrix[3][0] * v0 + matrix[3][1] * v1 + matrix[3][2] * v2 + matrix[3][3] * v3;
        });
    Kokkos::parallel_for(
        "two_target_dense_matrix_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dim * n_quads),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t row = g / n_quads;
            const std::uint64_t it = g % n_quads;
            const std::uint64_t col0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            const std::uint64_t col1 = col0 | lower_target_mask;
            const std::uint64_t col2 = col0 | upper_target_mask;
            const std::uint64_t col3 = col0 | target_mask;
            const Complex<Prec> v0 = dm._raw(row, col0);
            const Complex<Prec> v1 = dm._raw(row, col1);
            const Complex<Prec> v2 = dm._raw(row, col2);
            const Complex<Prec> v3 = dm._raw(row, col3);
            dm._raw(row, col0) = conj(matrix[0][0]) * v0 + conj(matrix[0][1]) * v1 +
                                 conj(matrix[0][2]) * v2 + conj(matrix[0][3]) * v3;
            dm._raw(row, col1) = conj(matrix[1][0]) * v0 + conj(matrix[1][1]) * v1 +
                                 conj(matrix[1][2]) * v2 + conj(matrix[1][3]) * v3;
            dm._raw(row, col2) = conj(matrix[2][0]) * v0 + conj(matrix[2][1]) * v1 +
                                 conj(matrix[2][2]) * v2 + conj(matrix[2][3]) * v3;
            dm._raw(row, col3) = conj(matrix[3][0]) * v0 + conj(matrix[3][1]) * v1 +
                                 conj(matrix[3][2]) * v2 + conj(matrix[3][3]) * v3;
        });
}

template <>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<Prec, Space>& matrix,
                             DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t dim = dm.dim();
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);
    const std::uint64_t outer_mask = ~target_mask & (dim - 1);
    const std::uint64_t active_dim = dim >> std::popcount(control_mask);
    auto update = DensityMatrix<Prec, Space>::uninitialized_state(dm.n_qubits());

    Kokkos::parallel_for(
        "multi_dense_matrix_gate_dm_initialize_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dim * dim),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t row = g / dim;
            const std::uint64_t col = g % dim;
            update._raw(row, col) =
                ((row & control_mask) == control_value_mask) ? Complex<Prec>{0} : dm._raw(row, col);
        });
    Kokkos::parallel_for(
        "multi_dense_matrix_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, active_dim * dim),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t active_row = g / dim;
            const std::uint64_t col = g % dim;
            const std::uint64_t matrix_row = active_row % matrix_dim;
            const std::uint64_t outer = active_row / matrix_dim;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_value_mask;
            const std::uint64_t dst_row =
                basis | insert_zero_at_mask_positions(matrix_row, outer_mask);
            Complex<Prec> sum = 0;
            for (std::uint64_t matrix_col = 0; matrix_col < matrix_dim; ++matrix_col) {
                const std::uint64_t src_row =
                    basis | insert_zero_at_mask_positions(matrix_col, outer_mask);
                sum += matrix(matrix_row, matrix_col) * dm._raw(src_row, col);
            }
            update._raw(dst_row, col) = sum;
        });
    std::swap(dm._raw, update._raw);

    Kokkos::parallel_for(
        "multi_dense_matrix_gate_dm_initialize_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dim * dim),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t row = g / dim;
            const std::uint64_t col = g % dim;
            update._raw(row, col) =
                ((col & control_mask) == control_value_mask) ? Complex<Prec>{0} : dm._raw(row, col);
        });
    Kokkos::parallel_for(
        "multi_dense_matrix_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dim * active_dim),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t row = g / active_dim;
            const std::uint64_t active_col = g % active_dim;
            const std::uint64_t matrix_row = active_col % matrix_dim;
            const std::uint64_t outer = active_col / matrix_dim;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_value_mask;
            const std::uint64_t dst_col =
                basis | insert_zero_at_mask_positions(matrix_row, outer_mask);
            Complex<Prec> sum = 0;
            for (std::uint64_t matrix_col = 0; matrix_col < matrix_dim; ++matrix_col) {
                const std::uint64_t src_col =
                    basis | insert_zero_at_mask_positions(matrix_col, outer_mask);
                sum += dm._raw(row, src_col) * conj(matrix(matrix_row, matrix_col));
            }
            update._raw(row, dst_col) = sum;
        });
    std::swap(dm._raw, update._raw);
}

}  // namespace scaluq::internal
