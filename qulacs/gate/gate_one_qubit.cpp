#include "gate_one_qubit.hpp"

#include "update_ops.hpp"

void Identity::update_quantum_state(StateVector& state_vector) const {
    i_gate(this->_target, state_vector);
}

void PauliX::update_quantum_state(StateVector& state_vector) const {
    x_gate(this->_target, state_vector);
}

void PauliY::update_quantum_state(StateVector& state_vector) const {
    y_gate(this->_target, state_vector);
}

void PauliZ::update_quantum_state(StateVector& state_vector) const {
    z_gate(this->_target, state_vector);
}

void Hadamard::update_quantum_state(StateVector& state_vector) const {
    h_gate(this->_target, state_vector);
}

void S::update_quantum_state(StateVector& state_vector) const {
    s_gate(this->_target, state_vector);
}

void Sdag::update_quantum_state(StateVector& state_vector) const {
    sdag_gate(this->_target, state_vector);
}

void T::update_quantum_state(StateVector& state_vector) const {
    t_gate(this->_target, state_vector);
}

void Tdag::update_quantum_state(StateVector& state_vector) const {
    tdag_gate(this->_target, state_vector);
}

void sqrtX::update_quantum_state(StateVector& state_vector) const {
    sqrtx_gate(this->_target, state_vector);
}

void sqrtXdag::update_quantum_state(StateVector& state_vector) const {
    sqrtxdag_gate(this->_target, state_vector);
}

void sqrtY::update_quantum_state(StateVector& state_vector) const {
    sqrty_gate(this->_target, state_vector);
}

void sqrtYdag::update_quantum_state(StateVector& state_vector) const {
    sqrtydag_gate(this->_target, state_vector);
}

void P0::update_quantum_state(StateVector& state_vector) const {
    p0_gate(this->_target, state_vector);
}

void P1::update_quantum_state(StateVector& state_vector) const {
    p1_gate(this->_target, state_vector);
}

