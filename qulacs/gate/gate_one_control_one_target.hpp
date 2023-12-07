#pragma once

#include <vector>

#include "gate.hpp"

namespace qulacs {
class CNOT : public QuantumGate {
    UINT _control, _target;

public:
    CNOT(UINT control, UINT target) : _control(control), _target(target){};
    void update_quantum_state(StateVector& state_vector) const override;
};

class CZ : public QuantumGate {
    UINT _control, _target;

public:
    CZ(UINT control, UINT target) : _control(control), _target(target){};
    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
