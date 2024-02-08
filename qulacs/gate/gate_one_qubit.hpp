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

class IGateImpl : public OneQubitGateBase {
public:
    IGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<IGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<IGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class XGateImpl : public OneQubitGateBase {
public:
    XGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<XGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<XGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class YGateImpl : public OneQubitGateBase {
public:
    YGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<YGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<YGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class ZGateImpl : public OneQubitGateBase {
public:
    ZGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<ZGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<ZGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class HGateImpl : public OneQubitGateBase {
public:
    HGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<HGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<HGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class SGateImpl : public OneQubitGateBase {
public:
    SGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<SGateImpl>(*this); }
    Gate get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class SdagGateImpl : public OneQubitGateBase {
public:
    SdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<SdagGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<SGateImpl>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class TGateImpl : public OneQubitGateBase {
public:
    TGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<TGateImpl>(*this); }
    Gate get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class TdagGateImpl : public OneQubitGateBase {
public:
    TdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<TdagGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<TGateImpl>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtXGateImpl : public OneQubitGateBase {
public:
    sqrtXGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<sqrtXGateImpl>(*this); }
    Gate get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtXdagGateImpl : public OneQubitGateBase {
public:
    sqrtXdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<sqrtXdagGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<sqrtXGateImpl>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtYGateImpl : public OneQubitGateBase {
public:
    sqrtYGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<sqrtYGateImpl>(*this); }
    Gate get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class sqrtYdagGateImpl : public OneQubitGateBase {
public:
    sqrtYdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<sqrtYdagGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<sqrtYGateImpl>(_target); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P0GateImpl : public OneQubitGateBase {
public:
    P0GateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<P0GateImpl>(*this); }
    Gate get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class P1GateImpl : public OneQubitGateBase {
public:
    P1GateImpl(UINT target) : OneQubitGateBase(target){};

    Gate copy() const override { return std::make_shared<P1GateImpl>(*this); }
    Gate get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RXGateImpl : public OneQubitRotationGateBase {
public:
    RXGateImpl(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    Gate copy() const override { return std::make_shared<RXGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<RXGateImpl>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RYGateImpl : public OneQubitRotationGateBase {
public:
    RYGateImpl(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    Gate copy() const override { return std::make_shared<RYGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<RYGateImpl>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class RZGateImpl : public OneQubitRotationGateBase {
public:
    RZGateImpl(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    Gate copy() const override { return std::make_shared<RZGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<RZGateImpl>(_target, -_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using IGate = internal::GatePtr<internal::IGateImpl>;
using XGate = internal::GatePtr<internal::XGateImpl>;
using YGate = internal::GatePtr<internal::YGateImpl>;
using ZGate = internal::GatePtr<internal::ZGateImpl>;
using HGate = internal::GatePtr<internal::HGateImpl>;
using SGate = internal::GatePtr<internal::SGateImpl>;
using SdagGate = internal::GatePtr<internal::SdagGateImpl>;
using TGate = internal::GatePtr<internal::TGateImpl>;
using TdagGate = internal::GatePtr<internal::TdagGateImpl>;
using sqrtXGate = internal::GatePtr<internal::sqrtXGateImpl>;
using sqrtXdagGate = internal::GatePtr<internal::sqrtXdagGateImpl>;
using sqrtYGate = internal::GatePtr<internal::sqrtYGateImpl>;
using sqrtYdagGate = internal::GatePtr<internal::sqrtYdagGateImpl>;
using P0Gate = internal::GatePtr<internal::P0GateImpl>;
using P1Gate = internal::GatePtr<internal::P1GateImpl>;
using RXGate = internal::GatePtr<internal::RXGateImpl>;
using RYGate = internal::GatePtr<internal::RYGateImpl>;
using RZGate = internal::GatePtr<internal::RZGateImpl>;
}  // namespace qulacs
