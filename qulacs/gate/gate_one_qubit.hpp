#pragma once

#include "gate.hpp"

namespace qulacs {
class I : public QuantumGate {
    UINT _target;

public:
    I(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<I>(*this); }
    Gate get_inverse() const override { return std::make_unique<I>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class X : public QuantumGate {
    UINT _target;

public:
    X(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<X>(*this); }
    Gate get_inverse() const override { return std::make_unique<X>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class Y : public QuantumGate {
    UINT _target;

public:
    Y(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<Y>(*this); }
    Gate get_inverse() const override { return std::make_unique<Y>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class Z : public QuantumGate {
    UINT _target;

public:
    Z(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<Z>(*this); }
    Gate get_inverse() const override { return std::make_unique<Z>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class H : public QuantumGate {
    UINT _target;

public:
    H(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<H>(*this); }
    Gate get_inverse() const override { return std::make_unique<H>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class S : public QuantumGate {
    UINT _target;

public:
    S(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<S>(*this); }
    Gate get_inverse() const override { return std::make_unique<Sdag>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class Sdag : public QuantumGate {
    UINT _target;

public:
    Sdag(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<Sdag>(*this); }
    Gate get_inverse() const override { return std::make_unique<S>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class T : public QuantumGate {
    UINT _target;

public:
    T(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<T>(*this); }
    Gate get_inverse() const override { return std::make_unique<Tdag>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class Tdag : public QuantumGate {
    UINT _target;

public:
    Tdag(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<Tdag>(*this); }
    Gate get_inverse() const override { return std::make_unique<T>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtX : public QuantumGate {
    UINT _target;

public:
    sqrtX(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<sqrtX>(*this); }
    Gate get_inverse() const override { return std::make_unique<sqrtXdag>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtXdag : public QuantumGate {
    UINT _target;

public:
    sqrtXdag(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<sqrtXdag>(*this); }
    Gate get_inverse() const override { return std::make_unique<sqrtX>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtY : public QuantumGate {
    UINT _target;

public:
    sqrtY(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<sqrtY>(*this); }
    Gate get_inverse() const override { return std::make_unique<sqrtYdag>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtYdag : public QuantumGate {
    UINT _target;

public:
    sqrtYdag(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<sqrtYdag>(*this); }
    Gate get_inverse() const override { return std::make_unique<sqrtY>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P0 : public QuantumGate {
    UINT _target;

public:
    P0(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<P0>(*this); }
    Gate get_inverse() const override {
        throw std::runtime_error("Projection gate hasn't inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P1 : public QuantumGate {
    UINT _target;

public:
    P1(UINT target) : _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<P1>(*this); }
    Gate get_inverse() const override {
        throw std::runtime_error("Projection gate hasn't inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RX : public QuantumGate {
    UINT _target;
    double _angle;

public:
    RX(UINT target, double angle) : _target(target), _angle(angle){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<RX>(*this); }
    Gate get_inverse() const override { return std::make_unique<RX>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RY : public QuantumGate {
    UINT _target;
    double _angle;

public:
    RY(UINT target, double angle) : _target(target), _angle(angle){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<RY>(*this); }
    Gate get_inverse() const override { return std::make_unique<RY>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RZ : public QuantumGate {
    UINT _target;
    double _angle;

public:
    RZ(UINT target, double angle) : _target(target), _angle(angle){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<RZ>(*this); }
    Gate get_inverse() const override { return std::make_unique<RZ>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
