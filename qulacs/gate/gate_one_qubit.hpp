#pragma once

#include "gate.hpp"

namespace qulacs {
class QuantumGateOneQubit : public QuantumGate {
protected:
    UINT _target;

public:
    QuantumGateOneQubit(UINT target) : _target(target){};

    UINT target() const { return _target; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };
};

class QuantumGateOneQubitRotation : public QuantumGateOneQubit {
protected:
    double _angle;

public:
    QuantumGateOneQubitRotation(UINT target, double angle)
        : QuantumGateOneQubit(target), _angle(angle){};

    double angle() const { return _angle; }
};

class I : public QuantumGateOneQubit {
public:
    I(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<I>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<I>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class X : public QuantumGateOneQubit {
public:
    X(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<X>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<X>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class Y : public QuantumGateOneQubit {
public:
    Y(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<Y>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<Y>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class Z : public QuantumGateOneQubit {
public:
    Z(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<Z>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<Z>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class H : public QuantumGateOneQubit {
public:
    H(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<H>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<H>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class S : public QuantumGateOneQubit {
public:
    S(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<S>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class Sdag : public QuantumGateOneQubit {
public:
    Sdag(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<Sdag>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<S>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class T : public QuantumGateOneQubit {
    UINT _target;

public:
    T(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<T>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class Tdag : public QuantumGateOneQubit {
public:
    Tdag(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<Tdag>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<T>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtX : public QuantumGateOneQubit {
public:
    sqrtX(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<sqrtX>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtXdag : public QuantumGateOneQubit {
public:
    sqrtXdag(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<sqrtXdag>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<sqrtX>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtY : public QuantumGateOneQubit {
public:
    sqrtY(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<sqrtY>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtYdag : public QuantumGateOneQubit {
public:
    sqrtYdag(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<sqrtYdag>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<sqrtY>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P0 : public QuantumGateOneQubit {
public:
    P0(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<P0>(*this); }
    GatePtr get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P1 : public QuantumGateOneQubit {
public:
    P1(UINT target) : QuantumGateOneQubit(target){};

    GatePtr copy() const override { return std::make_unique<P1>(*this); }
    GatePtr get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RX : public QuantumGateOneQubitRotation {
public:
    RX(UINT target, double angle) : QuantumGateOneQubitRotation(target, angle){};

    GatePtr copy() const override { return std::make_unique<RX>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<RX>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RY : public QuantumGateOneQubitRotation {
public:
    RY(UINT target, double angle) : QuantumGateOneQubitRotation(target, angle){};

    GatePtr copy() const override { return std::make_unique<RY>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<RY>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RZ : public QuantumGateOneQubitRotation {
public:
    RZ(UINT target, double angle) : QuantumGateOneQubitRotation(target, angle){};

    GatePtr copy() const override { return std::make_unique<RZ>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<RZ>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
