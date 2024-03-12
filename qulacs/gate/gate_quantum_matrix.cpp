#include "gate_quantum_matrix.hpp"

namespace qulacs {
namespace internal {
void U1GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u_gate(this->_target, this->_matrix, state_vector);
}

void U2GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u_gate(this->_target, this->_matrix, state_vector);
}

void U3GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u_gate(this->_target, this->_matrix, state_vector);
}
}  // namespace internal
}  // namespace qulacs
