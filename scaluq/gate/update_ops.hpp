
#pragma once

#include "../operator/pauli_operator.hpp"
#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
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
inline void check_qubit_within_bounds(const StateVectorBatched& states, UINT op_qubit) {
    if (op_qubit >= states.dim()) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVectorBatched& states): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}

void i_gate(StateVector& state);
void i_gate(StateVectorBatched& states);

void global_phase_gate(double angle, StateVector& state);
void global_phase_gate(double angle, StateVectorBatched& states);

void x_gate(UINT target_qubit_index, StateVector& state);
void x_gate(UINT target_qubit_index, StateVectorBatched& states);

void y_gate(UINT target_qubit_index, StateVector& state);
void y_gate(UINT target_qubit_index, StateVectorBatched& states);

void z_gate(UINT target_qubit_index, StateVector& state);
void z_gate(UINT target_qubit_index, StateVectorBatched& states);

void h_gate(UINT target_qubit_index, StateVector& state);
void h_gate(UINT target_qubit_index, StateVectorBatched& states);

void s_gate(UINT target_qubit_index, StateVector& state);
void s_gate(UINT target_qubit_index, StateVectorBatched& states);

void sdag_gate(UINT target_qubit_index, StateVector& state);
void sdag_gate(UINT target_qubit_index, StateVectorBatched& states);

void t_gate(UINT target_qubit_index, StateVector& state);
void t_gate(UINT target_qubit_index, StateVectorBatched& states);

void tdag_gate(UINT target_qubit_index, StateVector& state);
void tdag_gate(UINT target_qubit_index, StateVectorBatched& states);

void sqrtx_gate(UINT target_qubit_index, StateVector& state);
void sqrtx_gate(UINT target_qubit_index, StateVectorBatched& states);

void sqrtxdag_gate(UINT target_qubit_index, StateVector& state);
void sqrtxdag_gate(UINT target_qubit_index, StateVectorBatched& states);

void sqrty_gate(UINT target_qubit_index, StateVector& state);
void sqrty_gate(UINT target_qubit_index, StateVectorBatched& states);

void sqrtydag_gate(UINT target_qubit_index, StateVector& state);
void sqrtydag_gate(UINT target_qubit_index, StateVectorBatched& states);

void p0_gate(UINT target_qubit_index, StateVector& state);
void p0_gate(UINT target_qubit_index, StateVectorBatched& states);

void p1_gate(UINT target_qubit_index, StateVector& state);
void p1_gate(UINT target_qubit_index, StateVectorBatched& states);

void rx_gate(UINT target_qubit_index, double angle, StateVector& state);
void rx_gate(UINT target_qubit_index, double angle, StateVectorBatched& states);

void ry_gate(UINT target_qubit_index, double angle, StateVector& state);
void ry_gate(UINT target_qubit_index, double angle, StateVectorBatched& states);

void rz_gate(UINT target_qubit_index, double angle, StateVector& state);
void rz_gate(UINT target_qubit_index, double angle, StateVectorBatched& states);

void cx_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state);
void cx_gate(UINT control_qubit_index, UINT target_qubit_index, StateVectorBatched& states);

void cz_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state);
void cz_gate(UINT control_qubit_index, UINT target_qubit_index, StateVectorBatched& states);

matrix_2_2 get_IBMQ_matrix(double _theta, double _phi, double _lambda);

void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    const matrix_2_2& matrix,
                                    StateVector& state);
void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    const matrix_2_2& matrix,
                                    StateVectorBatched& states);

void double_qubit_dense_matrix_gate(UINT target0,
                                    UINT target1,
                                    const matrix_4_4& matrix,
                                    StateVector& state);
void double_qubit_dense_matrix_gate(UINT target0,
                                    UINT target1,
                                    const matrix_4_4& matrix,
                                    StateVectorBatched& states);

void u1_gate(UINT target_qubit_index, double lambda, StateVector& state);
void u1_gate(UINT target_qubit_index, double lambda, StateVectorBatched& states);

void u2_gate(UINT target_qubit_index, double phi, double lambda, StateVector& state);
void u2_gate(UINT target_qubit_index, double phi, double lambda, StateVectorBatched& states);

void u3_gate(UINT target_qubit_index, double theta, double phi, double lambda, StateVector& state);
void u3_gate(
    UINT target_qubit_index, double theta, double phi, double lambda, StateVectorBatched& states);

void swap_gate(UINT target1, UINT target2, StateVector& state);
void swap_gate(UINT target1, UINT target2, StateVectorBatched& states);

void fusedswap_gate(UINT target_qubit_index_0,
                    UINT target_qubit_index_1,
                    UINT block_size,
                    StateVector& state);
void fusedswap_gate(UINT target_qubit_index_0,
                    UINT target_qubit_index_1,
                    UINT block_size,
                    StateVectorBatched& states);

void pauli_gate(const PauliOperator& pauli, StateVector& state);
void pauli_gate(const PauliOperator& pauli, StateVectorBatched& states);

void pauli_rotation_gate(const PauliOperator& pauli, double angle, StateVector& state);
void pauli_rotation_gate(const PauliOperator& pauli, double angle, StateVectorBatched& states);
}  // namespace internal
}  // namespace scaluq
