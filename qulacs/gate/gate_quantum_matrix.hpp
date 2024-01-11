#pragma once

#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_one_qubit.hpp"
#include "update_ops.hpp"

namespace qulacs {
class U1 : public QuantumGateOneQubit {
    double _lambda;
    matrix_2_2 _matrix;

public:
    U1(UINT target, double lambda) : QuantumGateOneQubit(target), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(0, 0, lambda);
    };

    double lambda() const { return _lambda; }

    GatePtr copy() const override { return std::make_unique<U1>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<U1>(_target, -_lambda); }

    void update_quantum_state(StateVector& state_vector) const override;
};
class U2 : public QuantumGateOneQubit {
    double _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U2(UINT target, double phi, double lambda)
        : QuantumGateOneQubit(target), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(PI() / 2.0, phi, lambda);
    };

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    GatePtr copy() const override { return std::make_unique<U2>(*this); }
    GatePtr get_inverse() const override {
        return std::make_unique<U2>(_target, -_lambda - PI(), -_phi + PI());
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class U3 : public QuantumGateOneQubit {
    double _theta, _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U3(UINT target, double theta, double phi, double lambda)
        : QuantumGateOneQubit(target), _theta(theta), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(theta, phi, lambda);
    };

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    GatePtr copy() const override { return std::make_unique<U3>(*this); }
    GatePtr get_inverse() const override {
        return std::make_unique<U3>(_target, -_theta, -_lambda, -_phi);
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
