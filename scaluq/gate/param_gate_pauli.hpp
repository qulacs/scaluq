#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
class ParamPauliRotationGateImpl : public ParamGateBase {
    const PauliOperator _pauli;

public:
    ParamPauliRotationGateImpl(std::uint64_t control_mask,
                               const PauliOperator& pauli,
                               double param_coef = 1.)
        : ParamGateBase(vector_to_mask<false>(pauli.target_qubit_list()), control_mask, param_coef),
          _pauli(pauli) {}

    PauliOperator pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    ParamGate get_inverse() const override {
        return std::make_shared<const ParamPauliRotationGateImpl>(_control_mask, _pauli, -_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override {
        double angle = _pcoef * param;
        Complex true_angle = angle * this->_pauli.coef();
        ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
        StdComplex imag_unit(0, 1);
        mat = (StdComplex)Kokkos::cos(-true_angle / 2) *
                  ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * (StdComplex)Kokkos::sin(-true_angle / 2) * mat;
        return mat;
    }
    void update_quantum_state(StateVector& state_vector, double param) const override {
        pauli_rotation_gate(_control_mask, _pauli, _pcoef * param, state_vector);
    }
};
}  // namespace internal

using ParamPauliRotationGate = internal::ParamGatePtr<internal::ParamPauliRotationGateImpl>;
}  // namespace scaluq
