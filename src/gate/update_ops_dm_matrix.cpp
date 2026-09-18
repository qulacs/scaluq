#include "update_ops.hpp"

namespace scaluq::internal {

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
