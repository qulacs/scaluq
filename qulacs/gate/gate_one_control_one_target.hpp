#pragma once

#include <vector>

#include "gate.hpp"

namespace qulacs {
class QuantumGateOneControlOneTarget : public QuantumGate {
protected:
    UINT _control, _target;

public:
    QuantumGateOneControlOneTarget(UINT control, UINT target)
        : _control(control), _target(target){};

    UINT control() const { return _control; }
    UINT target() const { return _target; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {_control}; };
};

class CNOT : public QuantumGateOneControlOneTarget {
public:
    CNOT(UINT control, UINT target) : QuantumGateOneControlOneTarget(control, target) {}

    GatePtr copy() const override { return std::make_unique<CNOT>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<CNOT>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class CZ : public QuantumGateOneControlOneTarget {
public:
    CZ(UINT control, UINT target) : QuantumGateOneControlOneTarget(control, target) {}

    GatePtr copy() const override { return std::make_unique<CZ>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<CZ>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
