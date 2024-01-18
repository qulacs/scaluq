#pragma once

#include <vector>

#include "gate.hpp"

namespace qulacs {
class SWAP : public QuantumGate {
    UINT _target1, _target2;

public:
    SWAP(UINT target1, UINT target2) : _target1(target1), _target2(target2){};
    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
