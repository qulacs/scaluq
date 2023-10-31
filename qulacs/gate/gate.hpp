#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace qulacs {
class QuantumGate {
public:
    virtual void update_quantum_state(StateVector& state_vector) const = 0;
};
}  // namespace qulacs
