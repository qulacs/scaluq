#include "gate_quantum_matrix.hpp"

#include "update_ops.hpp"

namespace qulacs {
void U1::update_quantum_state(StateVector& state_vector) const {
    u_gate(this->_target, this->_matrix, state_vector);
}

void U2::update_quantum_state(StateVector& state_vector) const {
    u_gate(this->_target, this->_matrix, state_vector);
}

void U3::update_quantum_state(StateVector& state_vector) const {
    u_gate(this->_target, this->_matrix, state_vector);
}
} // namespace qulacs