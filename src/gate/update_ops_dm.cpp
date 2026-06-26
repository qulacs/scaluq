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

// Dense 1-qubit gate: ρ → U ρ U†.
//
// Uncontrolled (control_mask==0): single-pass block approach (Qulacs-style).
//   Precompute 4×4 extended matrix E[y*4+x] = U[y>>1][x>>1] * conj(U[y&1][x&1]).
//   Iterate over n_pairs² blocks; each block updates a complete 2×2 submatrix of ρ.
//
// Controlled: two-pass. Left pass applies U to all rows (for every column); right pass
//   applies U† to all columns (for every row). Cross-elements (ctrl active in row but
//   not col, or vice versa) are correctly handled because each pass iterates over all
//   columns/rows regardless of their control bits.
template <>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);

    if (control_mask == 0) {
        Kokkos::Array<Complex<Prec>, 16> ext;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                ext[y * 4 + x] = matrix[y >> 1][x >> 1] * conj(matrix[y & 1][x & 1]);

        Kokkos::parallel_for(
            "one_target_dense_matrix_gate_dm",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it_row = g / n_pairs;
                std::uint64_t it_col = g % n_pairs;
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it_row, target_mask) | control_value_mask;
                std::uint64_t row1 = row0 | target_mask;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it_col, target_mask) | control_value_mask;
                std::uint64_t col1 = col0 | target_mask;
                Complex<Prec> v00 = dm._raw(row0, col0), v01 = dm._raw(row0, col1);
                Complex<Prec> v10 = dm._raw(row1, col0), v11 = dm._raw(row1, col1);
                dm._raw(row0, col0) = ext[0]*v00 + ext[1]*v01 + ext[2]*v10 + ext[3]*v11;
                dm._raw(row0, col1) = ext[4]*v00 + ext[5]*v01 + ext[6]*v10 + ext[7]*v11;
                dm._raw(row1, col0) = ext[8]*v00 + ext[9]*v01 + ext[10]*v10 + ext[11]*v11;
                dm._raw(row1, col1) = ext[12]*v00 + ext[13]*v01 + ext[14]*v10 + ext[15]*v11;
            });
    } else {
        Kokkos::parallel_for(
            "one_target_dense_matrix_gate_dm_left",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it = g / dm.dim();
                std::uint64_t col = g % dm.dim();
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                std::uint64_t row1 = row0 | target_mask;
                Complex<Prec> val0 = dm._raw(row0, col);
                Complex<Prec> val1 = dm._raw(row1, col);
                dm._raw(row0, col) = matrix[0][0] * val0 + matrix[0][1] * val1;
                dm._raw(row1, col) = matrix[1][0] * val0 + matrix[1][1] * val1;
            });
        Kokkos::parallel_for(
            "one_target_dense_matrix_gate_dm_right",
            Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t row = g / n_pairs;
                std::uint64_t it = g % n_pairs;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                std::uint64_t col1 = col0 | target_mask;
                Complex<Prec> val0 = dm._raw(row, col0);
                Complex<Prec> val1 = dm._raw(row, col1);
                dm._raw(row, col0) = conj(matrix[0][0]) * val0 + conj(matrix[0][1]) * val1;
                dm._raw(row, col1) = conj(matrix[1][0]) * val0 + conj(matrix[1][1]) * val1;
            });
    }
}

// Diagonal gate: ρ → D ρ D†. (D ρ D†)_{ij} = d_{bit_q(i)} * conj(d_{bit_q(j)}) * ρ_{ij}.
// Uncontrolled: single-pass block; factor[y] = diag[y>>1]*conj(diag[y&1]) per block.
// Controlled: two-pass.
template <>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);

    if (control_mask == 0) {
        Kokkos::Array<Complex<Prec>, 4> factor;
        factor[0] = diag[0] * conj(diag[0]);
        factor[1] = diag[0] * conj(diag[1]);
        factor[2] = diag[1] * conj(diag[0]);
        factor[3] = diag[1] * conj(diag[1]);

        Kokkos::parallel_for(
            "one_target_diagonal_matrix_gate_dm",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it_row = g / n_pairs;
                std::uint64_t it_col = g % n_pairs;
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it_row, target_mask) | control_value_mask;
                std::uint64_t row1 = row0 | target_mask;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it_col, target_mask) | control_value_mask;
                std::uint64_t col1 = col0 | target_mask;
                dm._raw(row0, col0) *= factor[0];
                dm._raw(row0, col1) *= factor[1];
                dm._raw(row1, col0) *= factor[2];
                dm._raw(row1, col1) *= factor[3];
            });
    } else {
        Kokkos::parallel_for(
            "one_target_diagonal_matrix_gate_dm_left",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it = g / dm.dim();
                std::uint64_t col = g % dm.dim();
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                std::uint64_t row1 = row0 | target_mask;
                dm._raw(row0, col) *= diag[0];
                dm._raw(row1, col) *= diag[1];
            });
        Kokkos::parallel_for(
            "one_target_diagonal_matrix_gate_dm_right",
            Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t row = g / n_pairs;
                std::uint64_t it = g % n_pairs;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                std::uint64_t col1 = col0 | target_mask;
                dm._raw(row, col0) *= conj(diag[0]);
                dm._raw(row, col1) *= conj(diag[1]);
            });
    }
}

// Phase gate: only off-diagonal blocks (row_bit≠col_bit) change.
// Uncontrolled: single-pass block, 2 ops per block.
// Controlled: two-pass.
template <>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           DensityMatrix<Prec, Space>& dm) {
    Complex<Prec> phase_dag = conj(phase);
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);

    if (control_mask == 0) {
        Kokkos::parallel_for(
            "one_target_phase_gate_dm",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it_row = g / n_pairs;
                std::uint64_t it_col = g % n_pairs;
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it_row, target_mask) | control_value_mask;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it_col, target_mask) | control_value_mask;
                dm._raw(row0, col0 | target_mask) *= phase_dag;
                dm._raw(row0 | target_mask, col0) *= phase;
            });
    } else {
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
}

// GlobalPhase on DM:
//   Uncontrolled: U = e^{iφ}·I → ρ unchanged. No-op.
//   Controlled: two-pass (left multiplies active rows by coef, right by coef_dag).
template <>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       DensityMatrix<Prec, Space>& dm) {
    if (control_mask == 0) return;

    Complex<Prec> coef = internal::polar<Prec>(Float<Prec>{1}, angle);
    Complex<Prec> coef_dag = conj(coef);
    const std::uint64_t n_ctrl = dm.dim() >> std::popcount(control_mask);
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

// X gate: XρX† permutes ρ_{ij}→ρ_{flip(i),flip(j)}.
// Uncontrolled: block approach — two swaps per 2×2 block.
// Controlled: two-pass (swap rows then swap cols).
template <>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);

    if (control_mask == 0) {
        Kokkos::parallel_for(
            "x_gate_dm",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it_row = g / n_pairs;
                std::uint64_t it_col = g % n_pairs;
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it_row, target_mask) | control_value_mask;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it_col, target_mask) | control_value_mask;
                Kokkos::kokkos_swap(dm._raw(row0, col0),
                                    dm._raw(row0 | target_mask, col0 | target_mask));
                Kokkos::kokkos_swap(dm._raw(row0, col0 | target_mask),
                                    dm._raw(row0 | target_mask, col0));
            });
    } else {
        Kokkos::parallel_for(
            "x_gate_dm_left",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it = g / dm.dim();
                std::uint64_t col = g % dm.dim();
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                Kokkos::kokkos_swap(dm._raw(row0, col), dm._raw(row0 | target_mask, col));
            });
        Kokkos::parallel_for(
            "x_gate_dm_right",
            Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t row = g / n_pairs;
                std::uint64_t it = g % n_pairs;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                Kokkos::kokkos_swap(dm._raw(row, col0), dm._raw(row, col0 | target_mask));
            });
    }
}

// Y gate: YρY† with E derived from Y=[[0,-i],[i,0]].
// new_00=v11, new_01=-v10, new_10=-v01, new_11=v00.
// Uncontrolled: block approach. Controlled: two-pass.
template <>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t n_pairs = dm.dim() >> std::popcount(target_mask | control_mask);

    if (control_mask == 0) {
        Kokkos::parallel_for(
            "y_gate_dm",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it_row = g / n_pairs;
                std::uint64_t it_col = g % n_pairs;
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it_row, target_mask) | control_value_mask;
                std::uint64_t row1 = row0 | target_mask;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it_col, target_mask) | control_value_mask;
                std::uint64_t col1 = col0 | target_mask;
                Complex<Prec> v00 = dm._raw(row0, col0), v01 = dm._raw(row0, col1);
                Complex<Prec> v10 = dm._raw(row1, col0), v11 = dm._raw(row1, col1);
                dm._raw(row0, col0) =  v11;
                dm._raw(row0, col1) = -v10;
                dm._raw(row1, col0) = -v01;
                dm._raw(row1, col1) =  v00;
            });
    } else {
        Kokkos::parallel_for(
            "y_gate_dm_left",
            Kokkos::RangePolicy<SpaceType<Space>>(0, n_pairs * dm.dim()),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t it = g / dm.dim();
                std::uint64_t col = g % dm.dim();
                std::uint64_t row0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                std::uint64_t row1 = row0 | target_mask;
                Complex<Prec> val0 = dm._raw(row0, col);
                Complex<Prec> val1 = dm._raw(row1, col);
                dm._raw(row0, col) = Complex<Prec>(0, -1) * val1;
                dm._raw(row1, col) = Complex<Prec>(0, 1) * val0;
            });
        Kokkos::parallel_for(
            "y_gate_dm_right",
            Kokkos::RangePolicy<SpaceType<Space>>(0, dm.dim() * n_pairs),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t row = g / n_pairs;
                std::uint64_t it = g % n_pairs;
                std::uint64_t col0 =
                    insert_zero_at_mask_positions(it, control_mask | target_mask) |
                    control_value_mask;
                std::uint64_t col1 = col0 | target_mask;
                Complex<Prec> val0 = dm._raw(row, col0);
                Complex<Prec> val1 = dm._raw(row, col1);
                dm._raw(row, col0) = Complex<Prec>(0, 1) * val1;
                dm._raw(row, col1) = Complex<Prec>(0, -1) * val0;
            });
    }
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

// SWAP = SWAP†. Two-pass: swap lower/upper target rows, then swap lower/upper target cols.
template <>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               DensityMatrix<Prec, Space>& dm) {
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    const std::uint64_t n_quads = dm.dim() >> std::popcount(target_mask | control_mask);
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

// ECR: Hermitian 2-qubit gate. Precompute 16×16 extended matrix and apply as single-pass
// block update over n_quads² blocks (each updating a 4×4 submatrix of ρ).
// E[y*16+x] = ECR[y>>2][x>>2] * conj(ECR[y&3][x&3]).
template <>
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> s = static_cast<Float<Prec>>(INVERSE_SQRT2());
    using C = Complex<Prec>;
    // ECR in basis {00, ctrl-only, tgt-only, both}:
    const C ecr[4][4] = {
        {C(0), C(s),     C(0),      C(0, s)},
        {C(s), C(0),     C(0, -s),  C(0)},
        {C(0), C(0, s),  C(0),      C(s)},
        {C(0, -s), C(0), C(s),      C(0)}};

    Kokkos::Array<C, 256> ext;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x)
            ext[y * 16 + x] = ecr[y >> 2][x >> 2] * conj(ecr[y & 3][x & 3]);

    const std::uint64_t all_mask =
        physical_target_mask | physical_control_mask | control_mask;
    const std::uint64_t n_quads = dm.dim() >> std::popcount(all_mask);
    Kokkos::parallel_for(
        "ecr_gate_dm",
        Kokkos::RangePolicy<SpaceType<Space>>(0, n_quads * n_quads),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it_row = g / n_quads;
            std::uint64_t it_col = g % n_quads;
            std::uint64_t base_row =
                insert_zero_at_mask_positions(it_row, all_mask) | control_value_mask;
            std::uint64_t base_col =
                insert_zero_at_mask_positions(it_col, all_mask) | control_value_mask;

            std::uint64_t r[4] = {base_row,
                                  base_row | physical_control_mask,
                                  base_row | physical_target_mask,
                                  base_row | physical_control_mask | physical_target_mask};
            std::uint64_t c[4] = {base_col,
                                  base_col | physical_control_mask,
                                  base_col | physical_target_mask,
                                  base_col | physical_control_mask | physical_target_mask};

            C v[16];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    v[i * 4 + j] = dm._raw(r[i], c[j]);

            for (int y = 0; y < 16; ++y) {
                C sum(0);
                for (int x = 0; x < 16; ++x)
                    sum += ext[y * 16 + x] * v[x];
                dm._raw(r[y >> 2], c[y & 3]) = sum;
            }
        });
}

}  // namespace scaluq::internal
