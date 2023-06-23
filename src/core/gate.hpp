#pragma once

#include <cpusim/state_vector_cpu.hpp>

#include "types.hpp"

class QuantumGate {
public:
    virtual void update_quantum_state(StateVectorCpu& state_vector) const = 0;
};
