#pragma once

#include <vector>

#include "../operator/apply_pauli.hpp"
#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
class PPauliRotationGateImpl : public ParamGateBase {
    const PauliOperator _pauli;

public:
    PPauliRotationGateImpl(UINT control_mask, const PauliOperator& pauli, double pcoef = 1.)
        : ParamGateBase(vector_to_mask<false>(pauli.get_target_qubit_list()), control_mask, pcoef),
          _pauli(pauli) {}

    PauliOperator pauli() const { return _pauli; }
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }

    ParamGate get_inverse() const override {
        return std::make_shared<const PPauliRotationGateImpl>(_control_mask, _pauli, -_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override {
        double angle = _pcoef * param;
        Complex true_angle = angle * this->_pauli.get_coef();
        ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
        StdComplex imag_unit(0, 1);
        mat = Kokkos::cos(-true_angle / 2) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * Kokkos::sin(-true_angle / 2) * mat;
        return mat;
    }
    void update_quantum_state(StateVector& state_vector, double param) const override {
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
        apply_pauli_rotation(
            _control_mask, bit_flip_mask, phase_flip_mask, _pcoef * param, state_vector);
    }
};
}  // namespace internal

using PPauliRotationGate = internal::ParamGatePtr<internal::PPauliRotationGateImpl>;
}  // namespace scaluq
