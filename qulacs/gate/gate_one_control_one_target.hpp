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

class CNOTGateImpl : public OneControlOneTargetGateBase {
public:
    CNOTGateImpl(UINT control, UINT target) : OneControlOneTargetGateBase(control, target) {}

    Gate copy() const override { return std::make_shared<CNOTGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<CNOTGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(4, 4);
        mat << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class CZGateImpl : public OneControlOneTargetGateBase {
public:
    CZGateImpl(UINT control, UINT target) : OneControlOneTargetGateBase(control, target) {}

    Gate copy() const override { return std::make_shared<CZGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<CZGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(4, 4);
        mat << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using CNOTGate = internal::GatePtr<internal::CNOTGateImpl>;
using CZGate = internal::GatePtr<internal::CZGateImpl>;
}  // namespace qulacs
