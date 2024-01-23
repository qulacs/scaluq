#include "gate_two_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
void SWAP::update_quantum_state(StateVector& state_vector) const {
    swap_gate(this->_target1, this->_target2, state_vector);
}
}  // namespace qulacs
