#include "gate_pauli.hpp"

#include "../operator/pauli_operator.hpp"
#include "../tests/util/util.hpp"
#include "update_ops.hpp"

namespace qulacs {
namespace internal {

Gate PauliGateImpl::get_inverse() const { return std::make_shared<PauliGateImpl>(this->_pauli); }

Gate PauliRotationGateImpl::get_inverse() const {
    return std::make_shared<PauliRotationGateImpl>(this->_pauli, -(this->_angle));
}

void PauliGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_gate(this->_pauli, state_vector);
}

void PauliRotationGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_rotation_gate(this->_pauli, this->_angle, state_vector);
}

}  // namespace internal
}  // namespace qulacs
