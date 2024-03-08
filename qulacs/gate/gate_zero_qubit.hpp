#pragma once

#include "gate.hpp"

namespace qulacs {
namespace internal {
class ZeroQubitGateBase : public GateBase {
public:
    ZeroQubitGateBase(){};

    std::vector<UINT> get_target_qubit_list() const override { return {}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };
};

class IGateImpl : public ZeroQubitGateBase {
public:
    IGateImpl() : ZeroQubitGateBase(){};

    Gate copy() const override { return std::make_shared<IGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<IGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class GlobalPhaseGateImpl : public ZeroQubitGateBase {
protected:
    double _angle;

public:
    GlobalPhaseGateImpl(double angle) : ZeroQubitGateBase(), _angle(angle){};

    [[nodiscard]] double angle() const { return _angle; }

    Gate copy() const override { return std::make_shared<GlobalPhaseGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<GlobalPhaseGateImpl>(-_angle); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using IGate = internal::GatePtr<internal::IGateImpl>;
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl>;
}  // namespace qulacs
