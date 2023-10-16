#pragma once

#include "gate.hpp"

class PauliX : public QuantumGate {
    UINT _target;

public:
    PauliX(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class PauliY : public QuantumGate {
    UINT _target;

public:
    PauliY(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class PauliZ : public QuantumGate {
    UINT _target;

public:
    PauliZ(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};
