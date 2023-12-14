#pragma once

#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "update_ops.hpp"

namespace qulacs {
class U1 : public QuantumGate {
    UINT _target;
    double _lambda;
    std::array<Complex, 4> _matrix;

public:
    U1(UINT target, double lambda) : _target(target), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(0, 0, lambda);
    };

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<U1>(*this); }
    Gate get_inverse() const override { return std::make_unique<U1>(_target, -_lambda); }

    void update_quantum_state(StateVector& state_vector) const override;
};
class U2 : public QuantumGate {
    UINT _target;
    double _lambda, _phi;
    std::array<Complex, 4> _matrix;

public:
    U2(UINT target, double phi, double lambda) : _target(target), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(PI / 2.0, phi, lambda);
    };

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<U2>(*this); }
    Gate get_inverse() const override {
        return std::make_unique<U2>(_target, -_lambda - PI, -_phi + PI);
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class U3 : public QuantumGate {
    UINT _target;
    double _theta, _lambda, _phi;
    std::array<Complex, 4> _matrix;

public:
    U3(UINT target, double theta, double phi, double lambda)
        : _target(target), _theta(theta), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(theta, phi, lambda);
    };

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };

    Gate copy() const override { return std::make_unique<U3>(*this); }
    Gate get_inverse() const override {
        return std::make_unique<U3>(_target, -_theta, -_lambda, -_phi);
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
