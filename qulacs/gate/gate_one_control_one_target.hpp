#pragma once

#include <vector>

#include "gate.hpp"

namespace qulacs {
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

class CNOTGate : public OneControlOneTargetGateBase {
public:
    CNOTGate(UINT control, UINT target) : OneControlOneTargetGateBase(control, target) {}

    GatePtr copy() const override { return std::make_unique<CNOTGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<CNOTGate>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class CZGate : public OneControlOneTargetGateBase {
public:
    CZGate(UINT control, UINT target) : OneControlOneTargetGateBase(control, target) {}

    GatePtr copy() const override { return std::make_unique<CZGate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<CZGate>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal
}  // namespace qulacs
