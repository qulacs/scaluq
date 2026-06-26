#include "update_ops.hpp"

namespace scaluq::internal {

template <Precision Prec>
static Matrix2x2<Prec> ibmq_matrix(Float<Prec> _theta, Float<Prec> _phi, Float<Prec> _lambda) {
    Complex<Prec> exp_val1 = internal::exp(Complex<Prec>(0, _phi));
    Complex<Prec> exp_val2 = internal::exp(Complex<Prec>(0, _lambda));
    Complex<Prec> cos_val = internal::cos(_theta / Float<Prec>{2});
    Complex<Prec> sin_val = internal::sin(_theta / Float<Prec>{2});
    return {
        {{{cos_val, -exp_val2 * sin_val}}, {{exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val}}}};
}

template <>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);
    // Left pass: apply U to row index
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t row0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t row1 = row0 | target_mask;
            Complex<Prec> val0 = dm._raw(row0, col);
            Complex<Prec> val1 = dm._raw(row1, col);
            dm._raw(row0, col) = matrix[0][0] * val0 + matrix[0][1] * val1;
            dm._raw(row1, col) = matrix[1][0] * val0 + matrix[1][1] * val1;
        });
    // Right pass: apply U† to column index
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_pairs;
            std::uint64_t it = g % n_pairs;
            std::uint64_t col0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t col1 = col0 | target_mask;
            Complex<Prec> val0 = dm._raw(row, col0);
            Complex<Prec> val1 = dm._raw(row, col1);
            dm._raw(row, col0) = conj(matrix[0][0]) * val0 + conj(matrix[1][0]) * val1;
            dm._raw(row, col1) = conj(matrix[0][1]) * val0 + conj(matrix[1][1]) * val1;
        });
}

template <>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);
    // Left pass: apply D to row index
    Kokkos::parallel_for(
        "one_target_diagonal_matrix_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t row0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t row1 = row0 | target_mask;
            dm._raw(row0, col) *= diag[0];
            dm._raw(row1, col) *= diag[1];
        });
    // Right pass: apply D† to column index
    Kokkos::parallel_for(
        "one_target_diagonal_matrix_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_pairs;
            std::uint64_t it = g % n_pairs;
            std::uint64_t col0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t col1 = col0 | target_mask;
            dm._raw(row, col0) *= conj(diag[0]);
            dm._raw(row, col1) *= conj(diag[1]);
        });
}

template <>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);
    // Left pass: multiply |1> rows by phase
    Kokkos::parallel_for(
        "one_target_phase_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t row1 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) |
                control_value_mask | target_mask;
            dm._raw(row1, col) *= phase;
        });
    // Right pass: multiply |1> columns by conj(phase)
    Complex<Prec> phase_dag = conj(phase);
    Kokkos::parallel_for(
        "one_target_phase_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_pairs;
            std::uint64_t it = g % n_pairs;
            std::uint64_t col1 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) |
                control_value_mask | target_mask;
            dm._raw(row, col1) *= phase_dag;
        });
}

template <>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       DensityMatrix<Prec, Space>& dm) {
    Complex<Prec> coef = internal::polar<Prec>(Float<Prec>{1}, angle);
    Complex<Prec> coef_dag = conj(coef);
    const std::uint64_t n_ctrl = dm.dim() >> std::popcount(control_mask);
    // Left pass: multiply controlled rows by coef
    Kokkos::parallel_for(
        "global_phase_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_ctrl * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t row =
                insert_zero_at_mask_positions(it, control_mask) | control_value_mask;
            dm._raw(row, col) *= coef;
        });
    // Right pass: multiply controlled columns by conj(coef)
    Kokkos::parallel_for(
        "global_phase_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_ctrl),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_ctrl;
            std::uint64_t it = g % n_ctrl;
            std::uint64_t col =
                insert_zero_at_mask_positions(it, control_mask) | control_value_mask;
            dm._raw(row, col) *= coef_dag;
        });
}

template <>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);
    // Left pass: swap rows (X = X†)
    Kokkos::parallel_for(
        "x_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t row0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t row1 = row0 | target_mask;
            Kokkos::kokkos_swap(dm._raw(row0, col), dm._raw(row1, col));
        });
    // Right pass: swap cols
    Kokkos::parallel_for(
        "x_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_pairs;
            std::uint64_t it = g % n_pairs;
            std::uint64_t col0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t col1 = col0 | target_mask;
            Kokkos::kokkos_swap(dm._raw(row, col0), dm._raw(row, col1));
        });
}

template <>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);
    // Left pass: apply Y = [[0,-i],[i,0]] to rows
    Kokkos::parallel_for(
        "y_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t row0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t row1 = row0 | target_mask;
            Complex<Prec> val0 = dm._raw(row0, col);
            Complex<Prec> val1 = dm._raw(row1, col);
            dm._raw(row0, col) = Complex<Prec>(0, -1) * val1;
            dm._raw(row1, col) = Complex<Prec>(0, 1) * val0;
        });
    // Right pass: apply Y† = Y = [[0,-i],[i,0]] to cols
    Kokkos::parallel_for(
        "y_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_pairs;
            std::uint64_t it = g % n_pairs;
            std::uint64_t col0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t col1 = col0 | target_mask;
            Complex<Prec> val0 = dm._raw(row, col0);
            Complex<Prec> val1 = dm._raw(row, col1);
            dm._raw(row, col0) = Complex<Prec>(0, 1) * val1;
            dm._raw(row, col1) = Complex<Prec>(0, -1) * val0;
        });
}

template <>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(1, 0), Complex<Prec>(-1, 0)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, control_value_mask, diag, dm);
}

template <>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {
        {{{cosval, Complex<Prec>(0, -sinval)}}, {{Complex<Prec>(0, -sinval), cosval}}}};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, dm);
}

template <>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {{{{cosval, -sinval}}, {{sinval, cosval}}}};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, dm);
}

template <>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(cosval, -sinval), Complex<Prec>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, control_value_mask, diag, dm);
}

template <>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> lambda,
             DensityMatrix<Prec, Space>& dm) {
    Complex<Prec> exp_val = internal::exp(Complex<Prec>(0, lambda));
    DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(1, 0), exp_val};
    one_target_diagonal_matrix_gate(target_mask, control_mask, control_value_mask, diag, dm);
}

template <>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        control_value_mask,
        ibmq_matrix<Prec>(static_cast<Float<Prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        dm);
}

template <>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, ibmq_matrix<Prec>(theta, phi, lambda), dm);
}

template <>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               DensityMatrix<Prec, Space>& dm) {
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    const std::uint64_t n_quads = dm.dim() >> std::popcount(target_mask | control_mask);
    // Left pass: swap rows (SWAP = SWAP†)
    Kokkos::parallel_for(
        "swap_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_quads * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            Kokkos::kokkos_swap(dm._raw(basis | lower_target_mask, col),
                                dm._raw(basis | upper_target_mask, col));
        });
    // Right pass: swap cols
    Kokkos::parallel_for(
        "swap_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_quads),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_quads;
            std::uint64_t it = g % n_quads;
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            Kokkos::kokkos_swap(dm._raw(row, basis | lower_target_mask),
                                dm._raw(row, basis | upper_target_mask));
        });
}

template <>
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_quads =
        dm.dim() >>
        std::popcount(physical_target_mask | physical_control_mask | control_mask);
    // Left pass: apply ECR to rows
    Kokkos::parallel_for(
        "ecr_gate_dm_left",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_quads * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim();
            std::uint64_t col = g % dm.dim();
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(
                    it, physical_target_mask | physical_control_mask | control_mask) |
                control_value_mask;
            std::uint64_t basis_1 = basis_0 | physical_control_mask;
            std::uint64_t basis_2 = basis_0 | physical_target_mask;
            std::uint64_t basis_3 = basis_1 | physical_target_mask;
            Complex<Prec> v0 = dm._raw(basis_0, col);
            Complex<Prec> v1 = dm._raw(basis_1, col);
            Complex<Prec> v2 = dm._raw(basis_2, col);
            Complex<Prec> v3 = dm._raw(basis_3, col);
            const Float<Prec> inv_sqrt2 = static_cast<Float<Prec>>(INVERSE_SQRT2());
            dm._raw(basis_0, col) =
                (v1 + v3 * Complex<Prec>(0, 1)) * Complex<Prec>(inv_sqrt2, 0);
            dm._raw(basis_1, col) =
                (v0 + v2 * Complex<Prec>(0, -1)) * Complex<Prec>(inv_sqrt2, 0);
            dm._raw(basis_2, col) =
                (v1 * Complex<Prec>(0, 1) + v3) * Complex<Prec>(inv_sqrt2, 0);
            dm._raw(basis_3, col) =
                (v0 * Complex<Prec>(0, -1) + v2) * Complex<Prec>(inv_sqrt2, 0);
        });
    // Right pass: apply ECR† = ECR to cols (ECR is self-inverse)
    Kokkos::parallel_for(
        "ecr_gate_dm_right",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_quads),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n_quads;
            std::uint64_t it = g % n_quads;
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(
                    it, physical_target_mask | physical_control_mask | control_mask) |
                control_value_mask;
            std::uint64_t basis_1 = basis_0 | physical_control_mask;
            std::uint64_t basis_2 = basis_0 | physical_target_mask;
            std::uint64_t basis_3 = basis_1 | physical_target_mask;
            Complex<Prec> v0 = dm._raw(row, basis_0);
            Complex<Prec> v1 = dm._raw(row, basis_1);
            Complex<Prec> v2 = dm._raw(row, basis_2);
            Complex<Prec> v3 = dm._raw(row, basis_3);
            const Float<Prec> inv_sqrt2 = static_cast<Float<Prec>>(INVERSE_SQRT2());
            dm._raw(row, basis_0) =
                (v1 + v3 * Complex<Prec>(0, 1)) * Complex<Prec>(inv_sqrt2, 0);
            dm._raw(row, basis_1) =
                (v0 + v2 * Complex<Prec>(0, -1)) * Complex<Prec>(inv_sqrt2, 0);
            dm._raw(row, basis_2) =
                (v1 * Complex<Prec>(0, 1) + v3) * Complex<Prec>(inv_sqrt2, 0);
            dm._raw(row, basis_3) =
                (v0 * Complex<Prec>(0, -1) + v2) * Complex<Prec>(inv_sqrt2, 0);
        });
}

}  // namespace scaluq::internal
