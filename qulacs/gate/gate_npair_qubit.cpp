#include "gate_npair_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
namespace internal {
void FusedSWAPGateImpl::update_quantum_state(StateVector& state_vector) const {
    fusedswap_gate(this->_qubit_index1, this->_qubit_index2, this->_block_size, state_vector);
}
}  // namespace internal
}  // namespace qulacs
