#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "gate.hpp"

namespace qulacs {
namespace internal {
class PauliGateImpl : public GateBase {
    PauliOperator* _pauli;
    std::vector<UINT> _target_index_list, _pauli_id_list;

public:
    PauliGateImpl(PauliOperator* pauli) {
        _pauli = pauli;
        _target_index_list = _pauli->get_target_qubit_list();
        _pauli_id_list = _pauli->get_pauli_id_list();
    };

    std::vector<UINT> get_target_qubit_list() const override { return _target_index_list; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override { return std::make_shared<PauliGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<PauliGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class PauliRotationGateImpl : public GateBase {
    PauliOperator* _pauli;
    double _angle;
    std::vector<UINT> _target_index_list, _pauli_id_list;

public:
    PauliRotationGateImpl(PauliOperator* pauli, double angle) {
        _pauli = pauli;
        _angle = angle;
        _target_index_list = _pauli->get_target_qubit_list();
        _pauli_id_list = _pauli->get_pauli_id_list();
    };

    std::vector<UINT> get_target_qubit_list() const override { return _target_index_list; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override { return std::make_shared<PauliRotationGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<PauliRotationGateImpl>(*this); }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using PauliGate = internal::GatePtr<internal::PauliGateImpl>;
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl>;
}  // namespace qulacs
