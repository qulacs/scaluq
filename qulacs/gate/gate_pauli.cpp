#include "gate_pauli.hpp"

#include "../operator/pauli_operator.hpp"
#include "update_ops.hpp"

namespace qulacs {
namespace internal {

Gate PauliGateImpl::get_inverse() const {
    PauliOperator pauli(
        this->get_target_qubit_list(), this->get_pauli_id_list(), this->_pauli->get_coef());
    return std::make_shared<PauliGateImpl>(pauli);
}

Gate PauliRotationGateImpl::get_inverse() const {
    PauliOperator pauli(
        this->get_target_qubit_list(), this->get_pauli_id_list(), this->_pauli->get_coef());
    return std::make_shared<PauliRotationGateImpl>(pauli, -(this->_angle));
}

void PauliGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_gate(this->_pauli, state_vector);
}

void PauliRotationGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_rotation_gate(this->_pauli, this->_angle, state_vector);
}

}  // namespace internal
}  // namespace qulacs
