#pragma once

#include <vector>

#include "gate.hpp"

namespace scaluq {
namespace internal {
class OneControlOneTargetGateBase : public GateBase {
protected:
    UINT _control, _target;

public:
    OneControlOneTargetGateBase(UINT control, UINT target) : _control(control), _target(target){};

    UINT control() const { return _control; }
    UINT target() const { return _target; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {_control}; };
};

class CXGateImpl : public OneControlOneTargetGateBase {
public:
    CXGateImpl(UINT control, UINT target) : OneControlOneTargetGateBase(control, target) {}

    Gate copy() const override { return std::make_shared<CXGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<CXGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class CZGateImpl : public OneControlOneTargetGateBase {
public:
    CZGateImpl(UINT control, UINT target) : OneControlOneTargetGateBase(control, target) {}

    Gate copy() const override { return std::make_shared<CZGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<CZGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using CXGate = internal::GatePtr<internal::CXGateImpl>;
using CZGate = internal::GatePtr<internal::CZGateImpl>;
}  // namespace scaluq
