#pragma once

#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_one_qubit.hpp"
#include "update_ops.hpp"

namespace qulacs {
namespace internal {
class U1GateImpl : public OneQubitGateBase {
    double _lambda;
    matrix_2_2 _matrix;

public:
    U1GateImpl(UINT target, double lambda) : OneQubitGateBase(target), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(0, 0, lambda);
    };

    double lambda() const { return _lambda; }

    Gate copy() const override { return std::make_shared<U1GateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<U1GateImpl>(_target, -_lambda); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat = ComplexMatrix::Identity(2, 2);
        for (UINT i = 0; i < 2; ++i) {
            for (UINT j = 0; j < 2; ++j) {
                mat(i, j) = this->_matrix.val[i][j];
            }
        }
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
class U2GateImpl : public OneQubitGateBase {
    double _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U2GateImpl(UINT target, double phi, double lambda)
        : OneQubitGateBase(target), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(PI() / 2.0, phi, lambda);
    };

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate copy() const override { return std::make_shared<U2GateImpl>(*this); }
    Gate get_inverse() const override {
        return std::make_shared<U2GateImpl>(_target, -_lambda - PI(), -_phi + PI());
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class U3GateImpl : public OneQubitGateBase {
    double _theta, _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U3GateImpl(UINT target, double theta, double phi, double lambda)
        : OneQubitGateBase(target), _theta(theta), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(theta, phi, lambda);
    };

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate copy() const override { return std::make_shared<U3GateImpl>(*this); }
    Gate get_inverse() const override {
        return std::make_shared<U3GateImpl>(_target, -_theta, -_lambda, -_phi);
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using U1Gate = internal::GatePtr<internal::U1GateImpl>;
using U2Gate = internal::GatePtr<internal::U2GateImpl>;
using U3Gate = internal::GatePtr<internal::U3GateImpl>;
}  // namespace qulacs
