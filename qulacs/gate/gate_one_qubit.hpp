#pragma once

#include "gate.hpp"

namespace qulacs {
class I : public QuantumGate {
    UINT _target;

public:
    I(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class X : public QuantumGate {
    UINT _target;

public:
    X(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class Y : public QuantumGate {
    UINT _target;

public:
    Y(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class Z : public QuantumGate {
    UINT _target;

public:
    Z(UINT target) : _target(target){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class H : public QuantumGate {
    UINT _target;

public:
    H(UINT target) : _target(target){};

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

class RX : public QuantumGate {
    UINT _target;
    double _angle;

public:
    RX(UINT target, double angle) : _target(target), _angle(angle){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class RY : public QuantumGate {
    UINT _target;
    double _angle;

public:
    RY(UINT target, double angle) : _target(target), _angle(angle){};

    void update_quantum_state(StateVector& state_vector) const override;
};

class RZ : public QuantumGate {
    UINT _target;
    double _angle;

public:
    RZ(UINT target, double angle) : _target(target), _angle(angle){};

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
