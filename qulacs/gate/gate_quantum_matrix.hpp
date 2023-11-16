#pragma once

#include <vector>

#include "gate.hpp"
#include "update_ops.hpp"

namespace qulacs {
class U1 : public QuantumGate {
    UINT _target;
    double _lambda;
    std::array<Complex, 4> _matrix;

public:
    U1(UINT target, UINT lambda) : _target(target), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(0, 0, lambda);
    };
    void update_quantum_state(StateVector& state_vector) const override;
}
}  // namespace qulacs
