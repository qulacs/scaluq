#pragma once

#include "gate.hpp"

namespace qulacs {
namespace internal {
class OneQubitGateBase : public GateBase {
protected:
    UINT _target;

public:
    OneQubitGateBase(UINT target) : _target(target){};

    UINT target() const { return _target; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };
};

class OneQubitRotationGateBase : public OneQubitGateBase {
protected:
    double _angle;

public:
    OneQubitRotationGateBase(UINT target, double angle) : OneQubitGateBase(target), _angle(angle){};

    double angle() const { return _angle; }
};

class IGate : public OneQubitGateBase {
public:
    IGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<IGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<IGate>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class XGate : public OneQubitGateBase {
public:
    XGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<XGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<XGate>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class YGate : public OneQubitGateBase {
public:
    YGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<YGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<YGate>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class ZGate : public OneQubitGateBase {
public:
    ZGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<ZGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<ZGate>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class HGate : public OneQubitGateBase {
public:
    HGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<HGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<HGate>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class SGate : public OneQubitGateBase {
public:
    SGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<SGate>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class SdagGate : public OneQubitGateBase {
public:
    SdagGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<SdagGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<SGate>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class TGate : public OneQubitGateBase {
public:
    TGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<TGate>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class TdagGate : public OneQubitGateBase {
public:
    TdagGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<TdagGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<TGate>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtXGate : public OneQubitGateBase {
public:
    sqrtXGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<sqrtXGate>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtXdagGate : public OneQubitGateBase {
public:
    sqrtXdagGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<sqrtXdagGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<sqrtXGate>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtYGate : public OneQubitGateBase {
public:
    sqrtYGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<sqrtYGate>(*this); }
    GatePtr get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtYdagGate : public OneQubitGateBase {
public:
    sqrtYdagGate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<sqrtYdagGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<sqrtYGate>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P0Gate : public OneQubitGateBase {
public:
    P0Gate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<P0Gate>(*this); }
    GatePtr get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P1Gate : public OneQubitGateBase {
public:
    P1Gate(UINT target) : OneQubitGateBase(target){};

    GatePtr copy() const override { return std::make_unique<P1Gate>(*this); }
    GatePtr get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RXGate : public OneQubitRotationGateBase {
public:
    RXGate(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    GatePtr copy() const override { return std::make_unique<RXGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<RXGate>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RYGate : public OneQubitRotationGateBase {
public:
    RYGate(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    GatePtr copy() const override { return std::make_unique<RYGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<RYGate>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RZGate : public OneQubitRotationGateBase {
public:
    RZGate(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    GatePtr copy() const override { return std::make_unique<RZGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<RZGate>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal
}  // namespace qulacs
