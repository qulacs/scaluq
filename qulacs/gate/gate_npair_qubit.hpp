#pragma once

#include <cassert>

#include "gate.hpp"

namespace qulacs {
class FusedSWAP : public QuantumGate {
    UINT qubit_index1, qubit_index2, block_size;

public:
    FusedSWAP(UINT _qubit_index1, UINT _qubit_index2, UINT _block_size)
        : qubit_index1(_qubit_index1), qubit_index2(_qubit_index2), block_size(_block_size) {
        UINT upper_index = std::max(_qubit_index1, _qubit_index2);
        UINT lower_index = std::min(_qubit_index1, _qubit_index2);
        if (upper_index <= (lower_index + block_size - 1)) {
            throw std::runtime_error(
                "FusedSwap: upper index must be bigger than lower_index + block_size - 1");
        }
    };

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
