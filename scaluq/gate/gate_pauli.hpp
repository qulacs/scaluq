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
    PauliGateImpl(UINT control_mask, const PauliOperator& pauli)
        : GateBase(vector_to_mask(_pauli.get_target_qubit_list()), control_mask), _pauli(pauli) {}

    PauliOperator pauli() const { return _pauli; };
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }

    Gate get_inverse() const override { return shared_from_this(); }
    std::optional<ComplexMatrix> get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector& state_vector) const override {
        pauli_gate(_control_mask, _pauli, state_vector);
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
    
    PauliRotationGateImpl(UINT control_mask, const PauliOperator& pauli, double angle)
        : GateBase(vector_to_mask(_pauli.get_target_qubit_list()), control_mask),
          _pauli(pauli),
          _angle(angle) {}

    Gate get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl>(_control_mask, _pauli, -_angle);
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
        pauli_rotation_gate(_control_mask, _pauli, _angle, state_vector);
    }
};
}  // namespace internal

using PauliGate = internal::GatePtr<internal::PauliGateImpl>;
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl>;
}  // namespace scaluq
