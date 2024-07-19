#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "util/utility.hpp"

namespace scaluq {
namespace internal {
void x_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(UINT it) {
            UINT i = insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}
void y_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(UINT it) {
            UINT i = insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i] *= Complex(0, 1);
            state._raw[i | target_mask] *= Complex(0, -1);
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

void z_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(UINT it) {
            UINT i = insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= Complex(-1, 0);
        });
    Kokkos::fence();
}

void h_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(UINT it) {
            UINT i = insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Complex a = state._raw[i];
            Complex b = state._raw[i | target_mask];
            state._raw[i] = (a + b) * INVERSE_SQRT2();
            state._raw[i | target_mask] = (a - b) * INVERSE_SQRT2();
        });
    Kokkos::fence();
}

void single_qubit_phase_gate(UINT target_mask,
                             UINT control_mask,
                             Complex phase,
                             StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(UINT it) {
            UINT i = insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= phase;
        });
    Kokkos::fence();
}

void s_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_phase_gate(target_mask, control_mask, Complex(0, 1), state);
}

void sdag_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_phase_gate(target_mask, control_mask, Complex(0, -1), state);
}

void t_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_phase_gate(
        target_mask, control_mask, Complex(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}

void tdag_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_phase_gate(
        target_mask, control_mask, Complex(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}

void sqrtx_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX(), state);
}

void sqrtxdag_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX(), state);
}

void sqrty_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX(), state);
}

void sqrtydag_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX(), state);
}

void p0_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX(), state);
}

void p1_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    single_qubit_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX(), state);
}

void rx_gate(UINT target_mask, UINT control_mask, double angle, StateVector& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    matrix_2_2 matrix = {cosval, Complex(0, -sinval), Complex(0, -sinval), cosval};
    single_qubit_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

void ry_gate(UINT target_mask, UINT control_mask, double angle, StateVector& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    matrix_2_2 matrix = {cosval, -sinval, sinval, cosval};
    single_qubit_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

void single_qubit_diagonal_matrix_gate(UINT target_mask,
                                       UINT control_mask,
                                       const diagonal_matrix_2_2 diag,
                                       StateVector& state) {
    Kokkos::parallel_for(
        state.dim(),
        KOKKOS_LAMBDA(UINT it) { state._raw[it] *= diag.val[bool(it & target_mask)]; });
    Kokkos::fence();
}

void rz_gate(UINT target_mask, UINT control_mask, double angle, StateVector& state) {
    const double cosval = std::cos(angle / 2.);
    const double sinval = std::sin(angle / 2.);
    diagonal_matrix_2_2 diag = {Complex(cosval, -sinval), Complex(cosval, sinval)};
    single_qubit_diagonal_matrix_gate(target_mask, control_mask, diag, state);
}
}  // namespace internal
}  // namespace scaluq
