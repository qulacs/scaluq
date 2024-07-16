#pragma once

#include "gate.hpp"

namespace scaluq {
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

    Gate get_inverse() const override { return shared_from_this(); }
    std::optional<ComplexMatrix> get_matrix() const override {
        return ComplexMatrix::Identity(1, 1);
    }

    void update_quantum_state(StateVector& state_vector) const override { i_gate(state_vector); }
};

class GlobalPhaseGateImpl : public ZeroQubitGateBase {
protected:
    double _phase;

public:
    GlobalPhaseGateImpl(double phase) : ZeroQubitGateBase(), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    Gate get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl>(-_phase);
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        return ComplexMatrix::Identity(1, 1) * std::exp(std::complex<double>(0, _phase));
    }

    void update_quantum_state(StateVector& state_vector) const override {
        global_phase_gate(_phase, state_vector);
    }
};
}  // namespace internal

using IGate = internal::GatePtr<internal::IGateImpl>;
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl>;
}  // namespace scaluq
