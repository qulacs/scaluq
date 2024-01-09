
#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace qulacs {
void i_gate(UINT target_qubit_index, StateVector& state);

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

void cnot_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state);

void cz_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state);

matrix_2_2 get_IBMQ_matrix(double _theta, double _phi, double _lambda);

void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    const matrix_2_2& matrix,
                                    StateVector& state);

void u_gate(UINT target_qubit_index, const matrix_2_2& matrix, StateVector& state);
}  // namespace qulacs
