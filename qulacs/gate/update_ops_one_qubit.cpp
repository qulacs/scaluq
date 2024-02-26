#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "util/utility.hpp"

namespace qulacs {
void i_gate(UINT, StateVector&) {}

void x_gate(UINT target_qubit_index, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 1, KOKKOS_LAMBDA(const UINT& it) {
            UINT i = internal::insert_zero_to_basis_index(it, target_qubit_index);
            Kokkos::Experimental::swap(state._raw()[i],
                                       state._raw()[i | (1ULL << target_qubit_index)]);
        });
}

void y_gate(UINT target_qubit_index, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 1, KOKKOS_LAMBDA(const UINT& it) {
            UINT i = internal::insert_zero_to_basis_index(it, target_qubit_index);
            state._raw()[i] *= Complex(0, 1);
            state._raw()[i | (1ULL << target_qubit_index)] *= Complex(0, -1);
            Kokkos::Experimental::swap(state._raw()[i],
                                       state._raw()[i | (1ULL << target_qubit_index)]);
        });
}

void z_gate(UINT target_qubit_index, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 1, KOKKOS_LAMBDA(const UINT& it) {
            UINT i = internal::insert_zero_to_basis_index(it, target_qubit_index);
            state._raw()[i | (1ULL << target_qubit_index)] *= Complex(-1, 0);
        });
}

void h_gate(UINT target_qubit_index, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 1, KOKKOS_LAMBDA(const UINT& it) {
            UINT i = internal::insert_zero_to_basis_index(it, target_qubit_index);
            Complex a = state._raw()[i];
            Complex b = state._raw()[i | (1ULL << target_qubit_index)];
            state._raw()[i] = (a + b) * INVERSE_SQRT2();
            state._raw()[i | (1ULL << target_qubit_index)] = (a - b) * INVERSE_SQRT2();
        });
}

void single_qubit_phase_gate(UINT target_qubit_index, Complex phase, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 1, KOKKOS_LAMBDA(const UINT& it) {
            UINT i = internal::insert_zero_to_basis_index(it, target_qubit_index);
            state._raw()[i | (1ULL << target_qubit_index)] *= phase;
        });
}

void s_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(0, 1), state);
}

void sdag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(0, -1), state);
}

void t_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}

void tdag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}

void sqrtx_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_GATE_MATRIX(), state);
}

void sqrtxdag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_DAG_GATE_MATRIX(), state);
}

void sqrty_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_GATE_MATRIX(), state);
}

void sqrtydag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX(), state);
}

void p0_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, PROJ_0_MATRIX(), state);
}

void p1_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, PROJ_1_MATRIX(), state);
}

void rx_gate(UINT target_qubit_index, double angle, StateVector& state) {
    const double cosval = cos(angle / 2.);
    const double sinval = sin(angle / 2.);
    matrix_2_2 matrix = {cosval, Complex(0, -sinval), Complex(0, -sinval), cosval};
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state);
}

void ry_gate(UINT target_qubit_index, double angle, StateVector& state) {
    const double cosval = cos(angle / 2.);
    const double sinval = sin(angle / 2.);
    matrix_2_2 matrix = {cosval, -sinval, sinval, cosval};
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state);
}

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index,
                                       const diagonal_matrix_2_2 diag,
                                       StateVector& state) {
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(const UINT& it) {
            state._raw()[it] *= diag.val[(it >> target_qubit_index) & 1];
        });
}

void rz_gate(UINT target_qubit_index, double angle, StateVector& state) {
    const double cosval = cos(angle / 2.);
    const double sinval = sin(angle / 2.);
    diagonal_matrix_2_2 diag = {Complex(cosval, -sinval), Complex(cosval, sinval)};
    single_qubit_diagonal_matrix_gate(target_qubit_index, diag, state);
}
}  // namespace qulacs
