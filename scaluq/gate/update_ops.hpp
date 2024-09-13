
#pragma once

#include "../operator/pauli_operator.hpp"
#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace scaluq {
namespace internal {

STATE_VECTOR_TEMPLATE(FloatType, Space)
void i_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVector<FloatType, Space>& state) {}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void global_phase_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       double angle,
                       StateVector<FloatType, Space>& state) {
    Complex coef = Kokkos::polar(1., phase);
    Kokkos::parallel_for(
        state.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t i) {
            state._raw[insert_zero_at_mask_positions(i, control_mask) | control_mask] *= coef;
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVector<FloatType, Space>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVector<FloatType, Space>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i] *= Complex(0, 1);
            state._raw[i | target_mask] *= Complex(0, -1);
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVector<FloatType, Space>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= Complex(-1, 0);
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void h_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX(), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
inline void one_target_phase_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  Complex phase,
                                  StateVector<FloatType, Space>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= phase;
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void s_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVector<FloatType, Space>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex(0, 1), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void sdag_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVector<FloatType, Space>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex(0, -1), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void t_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVector<FloatType, Space>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void tdag_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVector<FloatType, Space>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void sqrtx_gate(std::uint64_t target_mask,
                std::uint64_t control_mask,
                StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX(), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void sqrtxdag_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX(), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void sqrty_gate(std::uint64_t target_mask,
                std::uint64_t control_mask,
                StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX(), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void sqrtydag_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX(), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void p0_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX(), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void p1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX(), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector<FloatType, Space>& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    Matrix2x2 matrix = {cosval, Complex(0, -sinval), Complex(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector<FloatType, Space>& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    Matrix2x2 matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector<FloatType, Space>& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    DiagonalMatrix2x2 diag = {Complex(cosval, -sinval), Complex(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
Matrix2x2 get_IBMQ_matrix(double _theta, double _phi, double _lambda);

STATE_VECTOR_TEMPLATE(FloatType, Space)
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2& matrix,
                                  StateVector<FloatType, Space>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4& matrix,
                                  StateVector<FloatType, Space>& state) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex val2 = state._raw[basis_2];
            Complex val3 = state._raw[basis_3];
            Complex res0 = matrix[0][0] * val0 + matrix[0][1] * val1 + matrix[0][2] * val2 +
                           matrix[0][3] * val3;
            Complex res1 = matrix[1][0] * val0 + matrix[1][1] * val1 + matrix[1][2] * val2 +
                           matrix[1][3] * val3;
            Complex res2 = matrix[2][0] * val0 + matrix[2][1] * val1 + matrix[2][2] * val2 +
                           matrix[2][3] * val3;
            Complex res3 = matrix[3][0] * val0 + matrix[3][1] * val1 + matrix[3][2] * val2 +
                           matrix[3][3] * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2& diag,
                                     StateVector<FloatType, Space>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            state._raw[basis] *= diag[0];
            state._raw[basis | target_mask] *= diag[1];
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double lambda,
             StateVector<FloatType, Space>& state) {
    Complex exp_val = Kokkos::exp(Complex(0, lambda));
    Kokkos::parallel_for(
        state.dim() >> (std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                internal::insert_zero_at_mask_positions(it, target_mask | control_mask) |
                control_mask;
            state._raw[i | target_mask] *= exp_val;
        });
    Kokkos::fence();
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double phi,
             double lambda,
             StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(Kokkos::numbers::pi / 2., phi, lambda), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double theta,
             double phi,
             double lambda,
             StateVector<FloatType, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(theta, phi, lambda), state);
}

STATE_VECTOR_TEMPLATE(FloatType, Space)
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVector<FloatType, Space>& state) {
    // '- target' is used for bit manipulation on unsigned type, not for its numerical meaning.
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[basis | lower_target_mask],
                                       state._raw[basis | upper_target_mask]);
        });
    Kokkos::fence();
}

}  // namespace internal
}  // namespace scaluq
