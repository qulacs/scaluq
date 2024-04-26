#pragma once

#include "../operator/pauli_operator.hpp"
#include "gate.hpp"
#include "update_ops.hpp"

namespace scaluq {

namespace internal {
class ParametricOneQubitGateBase : public ParametricGateBase {
protected:
    UINT _target;

public:
    ParametricOneQubitGateBase(UINT target) : _target(target){};

    UINT target() const { return _target; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };
};

class PRXGateImpl : public internal::ParametricOneQubitGateBase {
    PRXGateImpl(UINT target) : ParametricOneQubitGateBase(target) {}

    Gate copy() const override { return std::make_shared<PRXGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector, double parameter) const override {
        rx_gate(_target, parameter, state_vector);
    }
};

class PRYGateImpl : public internal::ParametricOneQubitGateBase {
    PRYGateImpl(UINT target) : ParametricOneQubitGateBase(target) {}

    Gate copy() const override { return std::make_shared<PRYGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector, double parameter) const override {
        ry_gate(_target, parameter, state_vector);
    }
};

class PRZGateImpl : public internal::ParametricOneQubitGateBase {
    PRZGateImpl(UINT target) : ParametricOneQubitGateBase(target) {}

    Gate copy() const override { return std::make_shared<PRZGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector, double parameter) const override {
        rx_gate(_target, parameter, state_vector);
    }
};

class PPauliRotationGateImpl : public ParametricGateBase {
    const PauliOperator _pauli;

public:
    PPauliRotationGateImpl(const PauliOperator& pauli) : _pauli(pauli) {}

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli.get_target_qubit_list();
    }
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override { return std::make_shared<PPauliRotationGateImpl>(_pauli); }
    void update_quantum_state(StateVector& state_vector, double parameter) const override {
        pauli_rotation_gate(_pauli, parameter, state_vector);
    }
};

}  // namespace internal

using PRXGate = internal::GatePtr<internal::PRXGateImpl>;
using PRYGate = internal::GatePtr<internal::PRYGateImpl>;
using PRZGate = internal::GatePtr<internal::PRZGateImpl>;
using PPauliRotationGate = internal::GatePtr<internal::PPauliRotationGateImpl>;

}  // namespace scaluq
