#include "gate_npair_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
void FusedSWAP::update_quantum_state(StateVector& state_vector) const {
    fusedswap_gate(this->qubit_index1, this->qubit_index2, this->block_size, state_vector);
}
}  // namespace qulacs
