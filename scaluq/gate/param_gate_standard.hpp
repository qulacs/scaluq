#pragma once

#include "../operator/pauli_operator.hpp"
#include "param_gate.hpp"
#include "update_ops.hpp"

namespace scaluq {

namespace internal {

class PRXGateImpl : public ParamGateBase {
public:
    using ParamGateBase::ParamGateBase;

    ParamGate get_inverse() const override {
        return std::make_shared<const PRXGateImpl>(_target_mask, _control_mask, -_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override {
        double angle = _pcoef * param;
        ComplexMatrix mat(2, 2);
        mat << std::cos(angle / 2), -1i * std::sin(angle / 2), -1i * std::sin(angle / 2),
            std::cos(angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        check_qubit_mask_within_bounds(state_vector);
        rx_gate(_target_mask, _control_mask, _pcoef * param, state_vector);
    }
};

class PRYGateImpl : public ParamGateBase {
public:
    using ParamGateBase::ParamGateBase;

    ParamGate get_inverse() const override {
        return std::make_shared<const PRYGateImpl>(_target_mask, _control_mask, -_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override {
        double angle = _pcoef * param;
        ComplexMatrix mat(2, 2);
        mat << std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        check_qubit_mask_within_bounds(state_vector);
        ry_gate(_target_mask, _control_mask, _pcoef * param, state_vector);
    }
};

class PRZGateImpl : public ParamGateBase {
public:
    using ParamGateBase::ParamGateBase;

    ParamGate get_inverse() const override {
        return std::make_shared<const PRZGateImpl>(_target_mask, _control_mask, -_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override {
        double angle = param * _pcoef;
        ComplexMatrix mat(2, 2);
        mat << std::exp(-0.5i * angle), 0, 0, std::exp(0.5i * angle);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        check_qubit_mask_within_bounds(state_vector);
        rz_gate(_target_mask, _control_mask, _pcoef * param, state_vector);
    }
};

}  // namespace internal

using PRXGate = internal::ParamGatePtr<internal::PRXGateImpl>;
using PRYGate = internal::ParamGatePtr<internal::PRYGateImpl>;
using PRZGate = internal::ParamGatePtr<internal::PRZGateImpl>;

}  // namespace scaluq
