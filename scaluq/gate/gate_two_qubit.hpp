#pragma once

#include <vector>

#include "gate.hpp"

namespace scaluq {
namespace internal {

class SwapGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return std::make_shared<SwapGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat = ComplexMatrix::Identity(1 << 2, 1 << 2);
        mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        swap_gate(_target_mask, _control_mask, state_vector);
    }
};
}  // namespace internal

using SwapGate = internal::GatePtr<internal::SwapGateImpl>;
}  // namespace scaluq
