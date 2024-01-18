#include "gate_one_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
namespace internal {
GatePtr SGate::get_inverse() const { return std::make_unique<SdagGate>(_target); }
GatePtr TGate::get_inverse() const { return std::make_unique<TdagGate>(_target); }
GatePtr sqrtXGate::get_inverse() const { return std::make_unique<sqrtXdagGate>(_target); }
GatePtr sqrtYGate::get_inverse() const { return std::make_unique<sqrtYdagGate>(_target); }

void IGate::update_quantum_state(StateVector& state_vector) const {
    i_gate(this->_target, state_vector);
}

void XGate::update_quantum_state(StateVector& state_vector) const {
    x_gate(this->_target, state_vector);
}

void YGate::update_quantum_state(StateVector& state_vector) const {
    y_gate(this->_target, state_vector);
}

void ZGate::update_quantum_state(StateVector& state_vector) const {
    z_gate(this->_target, state_vector);
}

void HGate::update_quantum_state(StateVector& state_vector) const {
    h_gate(this->_target, state_vector);
}

void SGate::update_quantum_state(StateVector& state_vector) const {
    s_gate(this->_target, state_vector);
}

void SdagGate::update_quantum_state(StateVector& state_vector) const {
    sdag_gate(this->_target, state_vector);
}

void TGate::update_quantum_state(StateVector& state_vector) const {
    t_gate(this->_target, state_vector);
}

void TdagGate::update_quantum_state(StateVector& state_vector) const {
    tdag_gate(this->_target, state_vector);
}

void sqrtXGate::update_quantum_state(StateVector& state_vector) const {
    sqrtx_gate(this->_target, state_vector);
}

void sqrtXdagGate::update_quantum_state(StateVector& state_vector) const {
    sqrtxdag_gate(this->_target, state_vector);
}

void sqrtYGate::update_quantum_state(StateVector& state_vector) const {
    sqrty_gate(this->_target, state_vector);
}

void sqrtYdagGate::update_quantum_state(StateVector& state_vector) const {
    sqrtydag_gate(this->_target, state_vector);
}

void P0Gate::update_quantum_state(StateVector& state_vector) const {
    p0_gate(this->_target, state_vector);
}

void P1Gate::update_quantum_state(StateVector& state_vector) const {
    p1_gate(this->_target, state_vector);
}

void RXGate::update_quantum_state(StateVector& state_vector) const {
    rx_gate(this->_target, this->_angle, state_vector);
}

void RYGate::update_quantum_state(StateVector& state_vector) const {
    ry_gate(this->_target, this->_angle, state_vector);
}

void RZGate::update_quantum_state(StateVector& state_vector) const {
    rz_gate(this->_target, this->_angle, state_vector);
}
}  // namespace internal
}  // namespace qulacs
