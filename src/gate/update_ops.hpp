
#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"

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
