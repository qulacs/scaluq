#include <core/gate_one_qubit.hpp>

#include "state_vector_cpu.hpp"
#include "update_ops.hpp"

void PauliX::update_quantum_state(StateVectorCpu& state_vector) const {
    x_gate(this->_target, state_vector);
}
