#pragma once

#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_one_qubit.hpp"
#include "update_ops.hpp"

namespace qulacs {
namespace internal {
class U1Gate : public OneQubitGateBase {
    double _lambda;
    matrix_2_2 _matrix;

public:
    U1Gate(UINT target, double lambda) : OneQubitGateBase(target), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(0, 0, lambda);
    };

    double lambda() const { return _lambda; }

    GatePtr copy() const override { return std::make_unique<U1Gate>(*this); }
    GatePtr get_inverse() const override { return std::make_unique<U1Gate>(_target, -_lambda); }

    void update_quantum_state(StateVector& state_vector) const override;
};
class U2Gate : public OneQubitGateBase {
    double _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U2Gate(UINT target, double phi, double lambda)
        : OneQubitGateBase(target), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(PI() / 2.0, phi, lambda);
    };

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    GatePtr copy() const override { return std::make_unique<U2Gate>(*this); }
    GatePtr get_inverse() const override {
        return std::make_unique<U2Gate>(_target, -_lambda - PI(), -_phi + PI());
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class U3Gate : public OneQubitGateBase {
    double _theta, _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U3Gate(UINT target, double theta, double phi, double lambda)
        : OneQubitGateBase(target), _theta(theta), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(theta, phi, lambda);
    };

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    GatePtr copy() const override { return std::make_unique<U3Gate>(*this); }
    GatePtr get_inverse() const override {
        return std::make_unique<U3Gate>(_target, -_theta, -_lambda, -_phi);
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal
}  // namespace qulacs
