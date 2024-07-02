#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
class PPauliRotationGateImpl : public ParamGateBase {
    const PauliOperator _pauli;

public:
    PPauliRotationGateImpl(const PauliOperator& pauli, double pcoef = 1.)
        : ParamGateBase(pcoef), _pauli(pauli) {}

    std::vector<UINT> get_target_qubit_list() const override {
        return _pauli.get_target_qubit_list();
    }
    std::vector<UINT> get_pauli_id_list() const { return _pauli.get_pauli_id_list(); }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    ParamGate get_inverse() const override {
        return std::make_shared<PPauliRotationGateImpl>(_pauli, -_pcoef);
    }
    std::optional<ComplexMatrix> get_matrix(double param) const override {
        double angle = _pcoef * param;
        ComplexMatrix mat = get_pauli_matrix(this->_pauli).value();
        StdComplex imag_unit(0, 1);
        mat = std::cos(-angle / 2) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * std::sin(-angle / 2) * mat;
        return mat;
    }
    void update_quantum_state(StateVector& state_vector, double param) const override {
        pauli_rotation_gate(_pauli, _pcoef * param, state_vector);
    }
};
}  // namespace internal

using PPauliRotationGate = internal::ParamGatePtr<internal::PPauliRotationGateImpl>;
}  // namespace scaluq
