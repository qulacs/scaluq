#include "gate_one_control_one_target.hpp"

#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void CXGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_control);
    check_qubit_within_bounds(state_vector, this->_target);
    cx_gate(this->_control, this->_target, state_vector);
}

void CZGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_control);
    check_qubit_within_bounds(state_vector, this->_target);
    cz_gate(this->_control, this->_target, state_vector);
}
}  // namespace internal
}  // namespace scaluq
