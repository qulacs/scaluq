#pragma once

#include "gate.hpp"

namespace scaluq {
namespace internal {

class IGateImpl : public GateBase {
public:
    IGateImpl() : GateBase(0, 0) {}

    Gate get_inverse() const override { return std::make_shared<IGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override {
        return ComplexMatrix::Identity(1, 1);
    }

    void update_quantum_state(StateVector& state_vector) const override {
        i_gate(_target_mask, _control_mask, state_vector);
    }
};

class GlobalPhaseGateImpl : public GateBase {
protected:
    double _phase;

public:
    GlobalPhaseGateImpl(UINT control_mask, double phase)
        : GateBase(0, control_mask), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    Gate get_inverse() const override {
        return std::make_shared<GlobalPhaseGateImpl>(_control_mask, -_phase);
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        return ComplexMatrix::Identity(1, 1) * std::exp(std::complex<double>(0, _phase));
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        global_phase_gate(_target_mask, _control_mask, _phase, state_vector);
    }
};
}  // namespace internal

using IGate = internal::GatePtr<internal::IGateImpl>;
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl>;
}  // namespace scaluq
