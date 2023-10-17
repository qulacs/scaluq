#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "update_ops.hpp"

void i_gate(UINT target_qubit_index, StateVector& state) {}

void x_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Kokkos::Experimental::swap(state[i], state[i | (1ULL << target_qubit_index)]);
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
            state[i] *= -im;
            state[i | (1ULL << target_qubit_index)] *= im;
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
            state[i | (1ULL << target_qubit_index)] *= -1;
        });
}

void h_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);  
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Complex a = state[i];
            Complex b = state[i | (1ULL << target_qubit_index)];
            state[i] = inv_sqrt2 * (a + b);
            state[i | (1ULL << target_qubit_index)] = inv_sqrt2 * (a - b);
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
    single_qubit_phase_gate(target_qubit_index, Complex(1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)), state);
}

void t_dag_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_phase_gate(target_qubit_index, Complex(1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0)), state);
}

void sqrtx_gate(UINT target_qubit_index, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, get_sqrt_x_matrix(), state);
}
