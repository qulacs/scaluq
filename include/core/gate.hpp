#pragma once

#include <core/types.hpp>
#include <cpusim/state_vector_cpu.hpp>

class QuantumGate {
public:
    virtual void update_quantum_state(StateVectorCpu& state_vector) const = 0;
};
