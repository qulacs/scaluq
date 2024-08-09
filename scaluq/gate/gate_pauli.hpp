#pragma once

#include <vector>

#include "../operator/apply_pauli.hpp"
#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
class PauliGateImpl : public GateBase {
    const PauliOperator _pauli;

public:
    PauliGateImpl(UINT control_mask, const PauliOperator& pauli)
        : GateBase(vector_to_mask<false>(pauli.get_target_qubit_list()), control_mask),
          _pauli(pauli) {}

    PauliOperator pauli() const { return _pauli; };
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }

    Gate get_inverse() const override { return shared_from_this(); }
    ComplexMatrix get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector& state_vector) const override {
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
        apply_pauli(_control_mask, bit_flip_mask, phase_flip_mask, _pauli.get_coef(), state_vector);
    }
};

class PauliRotationGateImpl : public GateBase {
    const PauliOperator _pauli;
    const double _angle;

public:
    PauliRotationGateImpl(UINT control_mask, const PauliOperator& pauli, double angle)
        : GateBase(vector_to_mask<false>(pauli.get_target_qubit_list()), control_mask),
          _pauli(pauli),
          _angle(angle) {}

    PauliOperator pauli() const { return _pauli; }
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }
    double angle() const { return _angle; }

    Gate get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl>(_control_mask, _pauli, -_angle);
    }

    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
        Complex true_angle = _angle * _pauli.get_coef();
        StdComplex imag_unit(0, 1);
        mat = Kokkos::cos(-true_angle / 2) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * Kokkos::sin(-true_angle / 2) * mat;
        return mat;
    }
    void update_quantum_state(StateVector& state_vector) const override {
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
        apply_pauli_rotation(
            _control_mask, bit_flip_mask, phase_flip_mask, _pauli.get_coef(), _angle, state_vector);
    }
};
}  // namespace internal

using PauliGate = internal::GatePtr<internal::PauliGateImpl>;
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl>;
}  // namespace scaluq
