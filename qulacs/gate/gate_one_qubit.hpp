#pragma once

#include "gate.hpp"

class PauliX : public QuantumGate {
    UINT _target;

public:
    PauliX(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};
