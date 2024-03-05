#include "gate_one_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
namespace internal {
Gate SGateImpl::get_inverse() const { return std::make_shared<SdagGateImpl>(_target); }
Gate TGateImpl::get_inverse() const { return std::make_shared<TdagGateImpl>(_target); }
Gate SqrtXGateImpl::get_inverse() const { return std::make_shared<SqrtXdagGateImpl>(_target); }
Gate SqrtYGateImpl::get_inverse() const { return std::make_shared<SqrtYdagGateImpl>(_target); }

void IGateImpl::update_quantum_state(StateVector& state_vector) const {
    i_gate(this->_target, state_vector);
}

void XGateImpl::update_quantum_state(StateVector& state_vector) const {
    x_gate(this->_target, state_vector);
}

void YGateImpl::update_quantum_state(StateVector& state_vector) const {
    y_gate(this->_target, state_vector);
}

void ZGateImpl::update_quantum_state(StateVector& state_vector) const {
    z_gate(this->_target, state_vector);
}

void HGateImpl::update_quantum_state(StateVector& state_vector) const {
    h_gate(this->_target, state_vector);
}

void SGateImpl::update_quantum_state(StateVector& state_vector) const {
    s_gate(this->_target, state_vector);
}

void SdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    sdag_gate(this->_target, state_vector);
}

void TGateImpl::update_quantum_state(StateVector& state_vector) const {
    t_gate(this->_target, state_vector);
}

void TdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    tdag_gate(this->_target, state_vector);
}

void SqrtXGateImpl::update_quantum_state(StateVector& state_vector) const {
    sqrtx_gate(this->_target, state_vector);
}

void SqrtXdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    sqrtxdag_gate(this->_target, state_vector);
}

void SqrtYGateImpl::update_quantum_state(StateVector& state_vector) const {
    sqrty_gate(this->_target, state_vector);
}

void SqrtYdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    sqrtydag_gate(this->_target, state_vector);
}

void P0GateImpl::update_quantum_state(StateVector& state_vector) const {
    p0_gate(this->_target, state_vector);
}

void P1GateImpl::update_quantum_state(StateVector& state_vector) const {
    p1_gate(this->_target, state_vector);
}

void RXGateImpl::update_quantum_state(StateVector& state_vector) const {
    rx_gate(this->_target, this->_angle, state_vector);
}

void RYGateImpl::update_quantum_state(StateVector& state_vector) const {
    ry_gate(this->_target, this->_angle, state_vector);
}

void RZGateImpl::update_quantum_state(StateVector& state_vector) const {
    rz_gate(this->_target, this->_angle, state_vector);
}
}  // namespace internal
}  // namespace qulacs
