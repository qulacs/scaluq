
#pragma once

#include "../operator/pauli_operator.hpp"
#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace scaluq {
namespace internal {
inline void check_qubit_within_bounds(const StateVector& state, UINT op_qubit) {
    if (op_qubit >= state.n_qubits()) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}

void i_gate(StateVector& state);

void global_phase_gate(double angle, StateVector& state);

void x_gate(UINT target_qubit_index, StateVector& state);

void y_gate(UINT target_qubit_index, StateVector& state);

void z_gate(UINT target_qubit_index, StateVector& state);

void h_gate(UINT target_qubit_index, StateVector& state);

void s_gate(UINT target_qubit_index, StateVector& state);

void sdag_gate(UINT target_qubit_index, StateVector& state);

void t_gate(UINT target_qubit_index, StateVector& state);

void tdag_gate(UINT target_qubit_index, StateVector& state);

void sqrtx_gate(UINT target_qubit_index, StateVector& state);

void sqrtxdag_gate(UINT target_qubit_index, StateVector& state);

void sqrty_gate(UINT target_qubit_index, StateVector& state);

void sqrtydag_gate(UINT target_qubit_index, StateVector& state);

void p0_gate(UINT target_qubit_index, StateVector& state);

void p1_gate(UINT target_qubit_index, StateVector& state);

void rx_gate(UINT target_qubit_index, double angle, StateVector& state);

void ry_gate(UINT target_qubit_index, double angle, StateVector& state);

void rz_gate(UINT target_qubit_index, double angle, StateVector& state);

void cx_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state);

void cz_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state);

matrix_2_2 get_IBMQ_matrix(double _theta, double _phi, double _lambda);

void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    const matrix_2_2& matrix,
                                    StateVector& state);

void double_qubit_dense_matrix_gate(UINT target0,
                                    UINT target1,
                                    const matrix_4_4& matrix,
                                    StateVector& state);

void u1_gate(UINT target_qubit_index, double lambda, StateVector& state);

void u2_gate(UINT target_qubit_index, double phi, double lambda, StateVector& state);

void u3_gate(UINT target_qubit_index, double theta, double phi, double lambda, StateVector& state);

void swap_gate(UINT target1, UINT target2, StateVector& state);

void fusedswap_gate(UINT target_qubit_index_0,
                    UINT target_qubit_index_1,
                    UINT block_size,
                    StateVector& state);

void pauli_gate(const PauliOperator& pauli, StateVector& state);

void pauli_rotation_gate(const PauliOperator& pauli, double angle, StateVector& state);
}  // namespace internal
}  // namespace scaluq
