#include "gate_one_qubit.hpp"

#include "update_ops.hpp"

Identity::Identity(UINT target) : _target(target){};

void Identity::update_quantum_state(StateVector& state_vector) const {
    i_gate(this->_target, state_vector);
}

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

Hadamard::Hadamard(UINT target) : _target(target){};

void Hadamard::update_quantum_state(StateVector& state_vector) const {
    h_gate(this->_target, state_vector);
}

S::S(UINT target) : _target(target){};

void S::update_quantum_state(StateVector& state_vector) const {
    s_gate(this->_target, state_vector);
}

Sdag::Sdag(UINT target) : _target(target){};

void Sdag::update_quantum_state(StateVector& state_vector) const {
    sdag_gate(this->_target, state_vector);
}

T::T(UINT target) : _target(target){};

void T::update_quantum_state(StateVector& state_vector) const {
    t_gate(this->_target, state_vector);
}

Tdag::Tdag(UINT target) : _target(target){};

void Tdag::update_quantum_state(StateVector& state_vector) const {
    tdag_gate(this->_target, state_vector);
}

sqrtX::sqrtX(UINT target) : _target(target){};

void sqrtX::update_quantum_state(StateVector& state_vector) const {
    sqrtx_gate(this->_target, state_vector);
}

sqrtXdag::sqrtXdag(UINT target) : _target(target){};

void sqrtXdag::update_quantum_state(StateVector& state_vector) const {
    sqrtxdag_gate(this->_target, state_vector);
}

