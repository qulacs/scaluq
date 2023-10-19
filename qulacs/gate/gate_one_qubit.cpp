#include "gate_one_qubit.hpp"

#include "update_ops.hpp"

void PauliX::update_quantum_state(StateVector& state_vector) const {
    x_gate(this->_target, state_vector);
}
