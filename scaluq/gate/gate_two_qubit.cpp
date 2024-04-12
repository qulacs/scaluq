#include "gate_two_qubit.hpp"

#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void SwapGateImpl::update_quantum_state(StateVector& state_vector) const {
    check_qubit_within_bounds(state_vector, this->_target1);
    check_qubit_within_bounds(state_vector, this->_target2);
    swap_gate(this->_target1, this->_target2, state_vector);
}
}  // namespace internal
}  // namespace scaluq
