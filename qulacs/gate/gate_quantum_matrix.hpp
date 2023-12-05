#pragma once

#include <vector>

#include "gate.hpp"
#include "update_ops.hpp"
#include "constant.hpp"

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
};
class U2 : public QuantumGate {
    UINT _target;
    double _lambda, _phi;
    std::array<Complex, 4> _matrix;

public:
    U2(UINT target, UINT lambda, UINT phi) : _target(target), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(PI / 2.0, phi, lambda);
    };
    void update_quantum_state(StateVector& state_vector) const override;
};

class U3 : public QuantumGate {
    UINT _target;
    double _theta, _lambda, _phi;
    std::array<Complex, 4> _matrix;

public:
    U3(UINT target, UINT theta, UINT lambda, UINT phi) : _target(target), _theta(theta), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(theta, phi, lambda);
    };
    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
