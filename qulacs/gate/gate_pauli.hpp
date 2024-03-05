#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "gate.hpp"

namespace qulacs {
namespace internal {
class PauliGateImpl : public GateBase {
    PauliOperator* _pauli;

public:
    PauliGateImpl(const PauliOperator& pauli) { *_pauli = pauli; };

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli->get_target_qubit_list();
    }
    std::vector<UINT> get_pauli_id_list() const { return _pauli->get_pauli_id_list(); }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override {
        PauliOperator pauli = *(new PauliOperator(*_pauli));
        return std::make_shared<PauliGateImpl>(pauli);
    }
    Gate get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};

class PauliRotationGateImpl : public GateBase {
    PauliOperator* _pauli;
    double _angle;

public:
    PauliRotationGateImpl(const PauliOperator& pauli, double angle) {
        *_pauli = pauli;
        _angle = angle;
    };

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli->get_target_qubit_list();
    }
    std::vector<UINT> get_pauli_id_list() const { return _pauli->get_pauli_id_list(); }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override {
        PauliOperator pauli = *(new PauliOperator(*_pauli));
        return std::make_shared<PauliRotationGateImpl>(pauli, _angle);
    }
    Gate get_inverse() const override;

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using PauliGate = internal::GatePtr<internal::PauliGateImpl>;
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl>;
}  // namespace qulacs
