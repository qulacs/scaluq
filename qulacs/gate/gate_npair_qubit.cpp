#include "gate_npair_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
namespace internal {
void FusedSwapGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_qubit_index1 + this->_block_size - 1);
    check_qubit_within_bounds(state_vector, this->_qubit_index2 + this->_block_size - 1);
    fusedswap_gate(this->_qubit_index1, this->_qubit_index2, this->_block_size, state_vector);
}
}  // namespace internal
}  // namespace qulacs
