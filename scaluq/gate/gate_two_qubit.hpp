#pragma once

#include <vector>

#include "gate.hpp"

namespace scaluq {
namespace internal {
class TwoQubitGateBase : public GateBase {
protected:
    UINT _target1, _target2;

public:
    TwoQubitGateBase(UINT target1, UINT target2) : _target1(target1), _target2(target2){};

    UINT target1() const { return _target1; }
    UINT target2() const { return _target2; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target1, _target2}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }
};

class SwapGateImpl : public TwoQubitGateBase {
public:
    SwapGateImpl(UINT target1, UINT target2) : TwoQubitGateBase(target1, target2) {}

    Gate copy() const override { return std::make_shared<SwapGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<SwapGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat = ComplexMatrix::Identity(1 << 2, 1 << 2);
        mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target1);
        check_qubit_within_bounds(state_vector, this->_target2);
        swap_gate(this->_target1, this->_target2, state_vector);
    }
};
}  // namespace internal

using SwapGate = internal::GatePtr<internal::SwapGateImpl>;
}  // namespace scaluq
