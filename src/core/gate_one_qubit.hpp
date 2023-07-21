#pragma once

#include <cpusim/state_vector_cpu.hpp>

#include "gate.hpp"
#include "types.hpp"

class PauliX : public QuantumGate {
    UINT _target;

public:
    PauliX(UINT target) : _target(target){};

    void update_quantum_state(StateVectorCpu& state_vector) const override;
};
