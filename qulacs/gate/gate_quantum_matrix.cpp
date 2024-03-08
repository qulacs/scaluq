#include "gate_quantum_matrix.hpp"

namespace qulacs {
namespace internal {
void U1GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u_gate(this->_target, this->_matrix, state_vector);
}

void U2GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u_gate(this->_target, this->_matrix, state_vector);
}

void U3GateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    u_gate(this->_target, this->_matrix, state_vector);
}

void OneQubitMatrixGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target);
    single_qubit_dense_matrix_gate(_target, _matrix, state_vector);
}

void TwoQubitMatrixGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target1);
    check_qubit_within_bounds(state_vector, this->_target2);
    double_qubit_dense_matrix_gate(_target1, _target2, _matrix, state_vector);
}
}  // namespace internal
}  // namespace qulacs
