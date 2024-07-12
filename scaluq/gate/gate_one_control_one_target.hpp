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

    Gate get_inverse() const override { return shared_from_this(); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(4, 4);
        mat << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_control);
        check_qubit_within_bounds(state_vector, this->_target);
        cx_gate(this->_control, this->_target, state_vector);
    }
};

class CZGateImpl : public OneControlOneTargetGateBase {
public:
    CZGateImpl(UINT control, UINT target) : OneControlOneTargetGateBase(control, target) {}

    Gate get_inverse() const override { return shared_from_this(); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(4, 4);
        mat << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_control);
        check_qubit_within_bounds(state_vector, this->_target);
        cz_gate(this->_control, this->_target, state_vector);
    }
};
}  // namespace internal

using CXGate = internal::GatePtr<internal::CXGateImpl>;
using CZGate = internal::GatePtr<internal::CZGateImpl>;
}  // namespace scaluq
