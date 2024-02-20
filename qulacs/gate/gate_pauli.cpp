#include "gate_pauli.hpp"

#include "../operator/pauli_operator.hpp"
#include "update_ops.hpp"

namespace qulacs {
namespace internal {

void PauliGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_gate(this->_pauli, state_vector);
}

void PauliRotationGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_rotation_gate(this->_pauli, this->_angle, state_vector);
}

}  // namespace internal
}  // namespace qulacs
