#include "gate_one_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
namespace internal {
Gate SGateImpl::get_inverse() const { return std::make_shared<SdagGateImpl>(_target); }
Gate TGateImpl::get_inverse() const { return std::make_shared<TdagGateImpl>(_target); }
Gate SqrtXGateImpl::get_inverse() const { return std::make_shared<SqrtXdagGateImpl>(_target); }
Gate SqrtYGateImpl::get_inverse() const { return std::make_shared<SqrtYdagGateImpl>(_target); }

void XGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    x_gate(this->_target, state_vector);
}

void YGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    y_gate(this->_target, state_vector);
}

void ZGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    z_gate(this->_target, state_vector);
}

void HGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    h_gate(this->_target, state_vector);
}

void SGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    s_gate(this->_target, state_vector);
}

void SdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    sdag_gate(this->_target, state_vector);
}

void TGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    t_gate(this->_target, state_vector);
}

void TdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    tdag_gate(this->_target, state_vector);
}

void SqrtXGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    sqrtx_gate(this->_target, state_vector);
}

void SqrtXdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    sqrtxdag_gate(this->_target, state_vector);
}

void SqrtYGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    sqrty_gate(this->_target, state_vector);
}

void SqrtYdagGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    sqrtydag_gate(this->_target, state_vector);
}

void P0GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    p0_gate(this->_target, state_vector);
}

void P1GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    p1_gate(this->_target, state_vector);
}

void RXGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    rx_gate(this->_target, this->_angle, state_vector);
}

void RYGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    ry_gate(this->_target, this->_angle, state_vector);
}

void RZGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    rz_gate(this->_target, this->_angle, state_vector);
}

void U1GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u1_gate(this->_target, this->_lambda, state_vector);
}

void U2GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u2_gate(this->_target, this->_phi, this->_lambda, state_vector);
}

void U3GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u3_gate(this->_target, this->_theta, this->_phi, this->_lambda, state_vector);
}
}  // namespace internal
}  // namespace qulacs
