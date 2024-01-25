#pragma once

#include <vector>

#include "gate.hpp"

namespace qulacs {
namespace internal {
class SWAPGateImpl : public GateBase {
    UINT _target1, _target2;

public:
    SWAPGateImpl(UINT target1, UINT target2) : _target1(target1), _target2(target2){};

    UINT target1() const { return _target1; }
    UINT target2() const { return _target2; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target1, _target2}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override { return std::make_shared<SWAPGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<SWAPGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using SWAPGate = internal::GatePtr<internal::SWAPGateImpl>;
}  // namespace qulacs
