#include "gate_pauli.hpp"

#include "update_ops.hpp"

namespace qulacs {
void Pauli::update_quantum_state(StateVector& state_vector) const {
    pauli_gate(this->target_qubit_index_list, this->pauli_id_list, state_vector);
}
void PauliRotation::update_quantum_state(StateVector& state_vector) const {
    pauli_rotation_gate(
        this->target_qubit_index_list, this->pauli_id_list, this->angle, state_vector);
}
}  // namespace qulacs
