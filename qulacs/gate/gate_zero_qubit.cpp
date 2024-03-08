#include "gate_zero_qubit.hpp"

#include "update_ops.hpp"

namespace qulacs {
namespace internal {
void IGateImpl::update_quantum_state(StateVector& state_vector) const { i_gate(state_vector); }

void GlobalPhaseGateImpl::update_quantum_state(StateVector& state_vector) const {
    global_phase_gate(_phase, state_vector);
}
}  // namespace internal
}  // namespace qulacs
