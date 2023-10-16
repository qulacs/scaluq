#include "gate_one_qubit.hpp"

#include "update_ops.hpp"

PauliX::PauliX(UINT target) : _target(target){};

void PauliX::update_quantum_state(StateVector& state_vector) const {
    x_gate(this->_target, state_vector);
}

PauliY::PauliY(UINT target) : _target(target){};

void PauliY::update_quantum_state(StateVector& state_vector) const {
    y_gate(this->_target, state_vector);
}

PauliZ::PauliZ(UINT target) : _target(target){};

void PauliZ::update_quantum_state(StateVector& state_vector) const {
    z_gate(this->_target, state_vector);
}
