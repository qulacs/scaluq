
#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"

void x_gate(UINT target_qubit_index, StateVector& state);

void y_gate(UINT target_qubit_index, StateVector& state);

void z_gate(UINT target_qubit_index, StateVector& state);
