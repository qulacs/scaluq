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
    PauliGateImpl(std::uint64_t control_mask, const PauliOperator& pauli)
        : GateBase(vector_to_mask<false>(pauli.get_target_qubit_list()), control_mask),
          _pauli(pauli) {}

    PauliOperator pauli() const { return _pauli; };
    std::vector<std::uint64_t> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }

    Gate get_inverse() const override { return shared_from_this(); }
    internal::ComplexMatrix get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector& state_vector) const override {
        pauli_gate(_control_mask, _pauli, state_vector);
    }
};

class PauliRotationGateImpl : public GateBase {
    const PauliOperator _pauli;
    const double _angle;

public:
    PauliRotationGateImpl(std::uint64_t control_mask, const PauliOperator& pauli, double angle)
        : GateBase(vector_to_mask<false>(pauli.get_target_qubit_list()), control_mask),
          _pauli(pauli),
          _angle(angle) {}

    PauliOperator pauli() const { return _pauli; }
    std::vector<std::uint64_t> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }
    double angle() const { return _angle; }

    Gate get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl>(_control_mask, _pauli, -_angle);
    }

    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
        Complex true_angle = _angle * _pauli.get_coef();
        StdComplex imag_unit(0, 1);
        mat = (StdComplex)Kokkos::cos(-true_angle / 2) *
                  internal::ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * (StdComplex)Kokkos::sin(-true_angle / 2) * mat;
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
