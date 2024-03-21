#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "gate.hpp"

namespace qulacs {
namespace internal {
class PauliGateImpl : public GateBase {
    const PauliOperator _pauli;

public:
    PauliGateImpl(const PauliOperator& pauli) : _pauli(pauli) {}

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli.get_target_qubit_list();
    }
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override { return std::make_shared<PauliGateImpl>(_pauli); }
    Gate get_inverse() const override;
    std::optional<ComplexMatrix> get_matrix() const override { return get_pauli_matrix(_pauli); }

    void update_quantum_state(StateVector& state_vector) const override;
};

class PauliRotationGateImpl : public GateBase {
    const PauliOperator _pauli;
    const double _angle;

public:
    PauliRotationGateImpl(const PauliOperator& pauli, double angle)
        : _pauli(pauli), _angle(angle) {}

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli.get_target_qubit_list();
    }
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate copy() const override { return std::make_shared<PauliRotationGateImpl>(_pauli, _angle); }
    Gate get_inverse() const override;
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat = get_pauli_matrix(_pauli).value();
        std::complex<double> imag_unit(0, 1);
        mat = cos(_angle / 2) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * sin(_angle / 2) * mat;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using PauliGate = internal::GatePtr<internal::PauliGateImpl>;
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl>;
}  // namespace qulacs
