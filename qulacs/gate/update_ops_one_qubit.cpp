#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

void i_gate(UINT target_qubit_index, StateVector& state) {}

void x_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    auto amplitudes = state.amplitudes_raw();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
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
    const Complex im(0, 1);
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            state[i] *= -1.i;
            state[i | (1ULL << target_qubit_index)] *= 1.i;
            Kokkos::Experimental::swap(state[i], state[i | (1ULL << target_qubit_index)]);
        });
}

void z_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            state[i | (1ULL << target_qubit_index)] *= -1.0;
        });
}

void h_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Complex a = state[i];
            Complex b = state[i | (1ULL << target_qubit_index)];
            state[i] = (a + b) * INVERSE_SQRT2;
            state[i | (1ULL << target_qubit_index)] = (a - b) * INVERSE_SQRT2;
        });
}

void single_qubit_phase_gate(
    UINT target_qubit_index, Complex phase, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            state[i | (1ULL << target_qubit_index)] *= phase;
        });
}

void s_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(0, 1), state);
}

void s_dag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(0, -1), state);
}

void t_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(1.0 / M_SQRT2, 1.0 / M_SQRT2), state);
}

void t_dag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(1.0 / M_SQRT2, -1.0 / M_SQRT2), state);
}

void single_qubit_dense_matrix_gate(
    UINT target_qubit_index, const Complex matrix[4], StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Complex cval_0 = state[i];
            Complex cval_1 = state[i | (1ULL << target_qubit_index)];
            state[i] = matrix[0] * cval_0 + matrix[1] * cval_1;
            state[i | (1ULL << target_qubit_index)] = matrix[2] * cval_0 + matrix[3] * cval_1;
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
