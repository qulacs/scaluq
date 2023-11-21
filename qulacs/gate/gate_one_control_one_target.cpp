#include "gate_one_control_one_target.hpp"

#include "update_ops_one_target_one_control.hpp"

namespace qulacs {
void CNOT::update_quantum_state(StateVector& state_vector) const {
    cnot_gate(this->_control, this->_target, state_vector);
}

void CZ::update_quantum_state(StateVector& state_vector) const {
    cz_gate(this->_control, this->_target, state_vector);
}
}  // namespace qulacs
