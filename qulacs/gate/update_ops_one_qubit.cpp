#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace qulacs {
void i_gate(UINT target_qubit_index, StateVector& state) {}

void x_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Kokkos::Experimental::swap(amplitudes[i], amplitudes[i | (1ULL << target_qubit_index)]);
        });
}

void y_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            amplitudes[i] *= Complex(0, 1);
            amplitudes[i | (1ULL << target_qubit_index)] *= Complex(0, -1);
            Kokkos::Experimental::swap(amplitudes[i], amplitudes[i | (1ULL << target_qubit_index)]);
        });
}

void z_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            amplitudes[i | (1ULL << target_qubit_index)] *= -1.0;
        });
}

void h_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Complex a = amplitudes[i];
            Complex b = amplitudes[i | (1ULL << target_qubit_index)];
            amplitudes[i] = (a + b) * INVERSE_SQRT2();
            amplitudes[i | (1ULL << target_qubit_index)] = (a - b) * INVERSE_SQRT2();
        });
}

void single_qubit_phase_gate(UINT target_qubit_index, Complex phase, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            amplitudes[i | (1ULL << target_qubit_index)] *= phase;
        });
}

void s_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(0, 1), state);
}

void sdag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(0, -1), state);
}

void t_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(1.0 / M_SQRT2, 1.0 / M_SQRT2), state);
}

void tdag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(1.0 / M_SQRT2, -1.0 / M_SQRT2), state);
}

void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    const matrix_2_2 matrix,
                                    StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Complex cval_0 = amplitudes[i];
            Complex cval_1 = amplitudes[i | (1ULL << target_qubit_index)];
            amplitudes[i] = matrix.val[0][0] * cval_0 + matrix.val[0][1] * cval_1;
            amplitudes[i | (1ULL << target_qubit_index)] =
                matrix.val[1][0] * cval_0 + matrix.val[1][1] * cval_1;
        });
}

void sqrtx_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_GATE_MATRIX, state);
}

void sqrtxdag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state);
}

void sqrty_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_GATE_MATRIX, state);
}

void sqrtydag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state);
}

void p0_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, PROJ_0_MATRIX, state);
}

void p1_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, PROJ_1_MATRIX, state);
}

void rx_gate(UINT target_qubit_index, double angle, StateVector& state) {
    const double cosval = cos(angle / 2.);
    const double sinval = sin(angle / 2.);
    matrix_2_2 matrix = {cosval, -Complex(0, sinval), -Complex(0, sinval), cosval};
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state);
}

void ry_gate(UINT target_qubit_index, double angle, StateVector& state) {
    const double cosval = cos(angle / 2.);
    const double sinval = sin(angle / 2.);
    matrix_2_2 matrix = {cosval, -sinval, sinval, cosval};
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state);
}

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index,
                                       const Complex diagonal_matrix[2],
                                       StateVector& state) {
    UINT mask = 1ULL << target_qubit_index;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(const UINT& it) {
            int bitval = ((it & mask) != 0);
            amplitudes[it] *= diagonal_matrix[bitval];
        });
}

void rz_gate(UINT target_qubit_index, double angle, StateVector& state) {
    const double cosval = cos(angle / 2.);
    const double sinval = sin(angle / 2.);
    Complex diagonal_matrix[2] = {cosval - Complex(0, sinval), cosval + Complex(0, sinval)};
    single_qubit_diagonal_matrix_gate(target_qubit_index, diagonal_matrix, state);
}
}  // namespace qulacs
