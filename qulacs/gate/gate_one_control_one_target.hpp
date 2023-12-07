#pragma once

#include <vector>

#include "gate.hpp"

namespace qulacs {
class CNOT : public QuantumGate {
    UINT _control, _target;

public:
    CNOT(UINT control, UINT target) : _control(control), _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {_control}; };

    Gate copy() const override { return std::make_unique<CNOT>(*this); }
    Gate get_inverse() const override { return std::make_unique<CNOT>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class CZ : public QuantumGate {
    UINT _control, _target;

public:
    CZ(UINT control, UINT target) : _control(control), _target(target){};

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {_control}; };

    Gate copy() const override { return std::make_unique<CZ>(*this); }
    Gate get_inverse() const override { return std::make_unique<CZ>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
