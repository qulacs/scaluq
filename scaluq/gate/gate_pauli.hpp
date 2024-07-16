#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
class PauliGateImpl : public GateBase {
    const PauliOperator _pauli;

public:
    PauliGateImpl(const PauliOperator& pauli) : _pauli(pauli) {}

    PauliOperator pauli() const { return _pauli; };
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli.get_target_qubit_list();
    }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate get_inverse() const override { return shared_from_this(); }
    std::optional<ComplexMatrix> get_matrix() const override {
        return get_pauli_matrix(this->_pauli);
    }

    void update_quantum_state(StateVector& state_vector) const override {
        pauli_gate(this->_pauli, state_vector);
    }
};

class PauliRotationGateImpl : public GateBase {
    const PauliOperator _pauli;
    const double _angle;

public:
    PauliRotationGateImpl(const PauliOperator& pauli, double angle)
        : _pauli(pauli), _angle(angle) {}

    PauliOperator pauli() const { return _pauli; }
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }
    double angle() const { return _angle; }

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli.get_target_qubit_list();
    }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    Gate get_inverse() const override {
        return std::make_shared<PauliRotationGateImpl>(this->_pauli, -(this->_angle));
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
        Complex true_angle = _angle * _pauli.get_coef();
        StdComplex imag_unit(0, 1);
        mat = Kokkos::cos(-true_angle / 2) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * Kokkos::sin(-true_angle / 2) * mat;
        return mat;
    }
    void update_quantum_state(StateVector& state_vector) const override {
        pauli_rotation_gate(this->_pauli, this->_angle, state_vector);
    }
};
}  // namespace internal

using PauliGate = internal::GatePtr<internal::PauliGateImpl>;
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl>;
}  // namespace scaluq
