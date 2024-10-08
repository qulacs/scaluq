#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void i_gate(std::uint64_t, std::uint64_t, StateVector&) {}

void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       double phase,
                       StateVector& state) {
    Complex coef = Kokkos::polar(1., phase);
    Kokkos::parallel_for(
        state.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t i) {
            state._raw[insert_zero_at_mask_positions(i, control_mask) | control_mask] *= coef;
        });
    Kokkos::fence();
}

void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
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

void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= Complex(-1, 0);
        });
    Kokkos::fence();
}

void h_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX(), state);
}

void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex phase,
                           StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= phase;
        });
    Kokkos::fence();
}

void s_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_phase_gate(target_mask, control_mask, Complex(0, 1), state);
}

void sdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_phase_gate(target_mask, control_mask, Complex(0, -1), state);
}

void t_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}

void tdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}

void sqrtx_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX(), state);
}

void sqrtxdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX(), state);
}

void sqrty_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX(), state);
}

void sqrtydag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX(), state);
}

void p0_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX(), state);
}

void p1_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX(), state);
}

void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    Matrix2x2 matrix = {cosval, Complex(0, -sinval), Complex(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    Matrix2x2 matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2& diag,
                                     StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            state._raw[basis] *= diag[0];
            state._raw[basis | target_mask] *= diag[1];
        });
    Kokkos::fence();
}

void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    DiagonalMatrix2x2 diag = {Complex(cosval, -sinval), Complex(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, state);
}

Matrix2x2 get_IBMQ_matrix(double theta, double phi, double lambda) {
    Complex exp_val1 = Kokkos::exp(Complex(0, phi));
    Complex exp_val2 = Kokkos::exp(Complex(0, lambda));
    Complex cos_val = Kokkos::cos(theta / 2.);
    Complex sin_val = Kokkos::sin(theta / 2.);
    return {cos_val, -exp_val2 * sin_val, exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val};
}

void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double lambda,
             StateVector& state) {
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

void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double phi,
             double lambda,
             StateVector& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(Kokkos::numbers::pi / 2., phi, lambda), state);
}

void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double theta,
             double phi,
             double lambda,
             StateVector& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(theta, phi, lambda), state);
}

void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state) {
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
