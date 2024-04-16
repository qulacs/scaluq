#include "gate_pauli.hpp"

#include "../operator/pauli_operator.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {

Gate PauliGateImpl::get_inverse() const { return std::make_shared<PauliGateImpl>(this->_pauli); }

Gate PauliRotationGateImpl::get_inverse() const {
    return std::make_shared<PauliRotationGateImpl>(this->_pauli, -(this->_angle));
}

std::optional<ComplexMatrix> PauliGateImpl::get_matrix() const {
    return get_pauli_matrix(this->_pauli);
}

std::optional<ComplexMatrix> PauliRotationGateImpl::get_matrix() const {
    ComplexMatrix mat = get_pauli_matrix(this->_pauli).value();
    StdComplex imag_unit(0, 1);
    mat = cos(-_angle / 2) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
          imag_unit * sin(-_angle / 2) * mat;
    return mat;
}

void PauliGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_gate(this->_pauli, state_vector);
}

void PauliRotationGateImpl::update_quantum_state(StateVector& state_vector) const {
    pauli_rotation_gate(this->_pauli, this->_angle, state_vector);
}

}  // namespace internal
}  // namespace scaluq
