#pragma once

#include "gate.hpp"

class Identity : public QuantumGate {
    UINT _target;

public:
    Identity(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

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

class Hadamard : public QuantumGate {
    UINT _target;

public:
    Hadamard(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class S : public QuantumGate {
    UINT _target;

public:
    S(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class Sdag : public QuantumGate {
    UINT _target;

public:
    Sdag(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class T : public QuantumGate {
    UINT _target;

public:
    T(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class Tdag : public QuantumGate {
    UINT _target;

public:
    Tdag(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtX : public QuantumGate {
    UINT _target;

public:
    sqrtX(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtXdag : public QuantumGate {
    UINT _target;

public:
    sqrtXdag(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtY : public QuantumGate {
    UINT _target;

public:
    sqrtY(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtYdag : public QuantumGate {
    UINT _target;

public:
    sqrtYdag(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class P0 : public QuantumGate {
    UINT _target;

public:
    P0(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class P1 : public QuantumGate {
    UINT _target;

public:
    P1(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};



