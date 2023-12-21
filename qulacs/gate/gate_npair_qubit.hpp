#pragma once

#include "gate.hpp"

namespace qulacs {
class FusedSWAP : public QuantumGate {
    UINT qubit_index1, qubit_index2, block_size;

public:
    FusedSWAP(UINT _qubit_index1, UINT _qubit_index2, UINT _block_size)
        : qubit_index1(_qubit_index1), qubit_index2(_qubit_index2), block_size(_block_size){};

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
