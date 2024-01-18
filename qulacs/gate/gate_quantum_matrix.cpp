#include "gate_quantum_matrix.hpp"

namespace qulacs {
namespace internal {
void U1Gate::update_quantum_state(StateVector& state_vector) const {
    u_gate(this->_target, this->_matrix, state_vector);
}

void U2Gate::update_quantum_state(StateVector& state_vector) const {
    u_gate(this->_target, this->_matrix, state_vector);
}

void U3Gate::update_quantum_state(StateVector& state_vector) const {
    u_gate(this->_target, this->_matrix, state_vector);
}
}  // namespace internal
}  // namespace qulacs
