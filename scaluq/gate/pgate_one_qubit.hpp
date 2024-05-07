#pragma once

#include "../operator/pauli_operator.hpp"
#include "pgate.hpp"
#include "update_ops.hpp"

namespace scaluq {

namespace internal {
class POneQubitGateBase : public PGateBase {
protected:
    UINT _target;

public:
    POneQubitGateBase(UINT target, double pcoef = 1.) : PGateBase(pcoef), _target(target){};

    UINT target() const { return _target; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };
};

class PRXGateImpl : public POneQubitGateBase {
public:
    PRXGateImpl(UINT target, double pcoef = 1.) : POneQubitGateBase(target, pcoef){};

    PGate copy() const override { return std::make_shared<PRXGateImpl>(*this); }
    PGate get_inverse() const override { return std::make_shared<PRXGateImpl>(_target, -_pcoef); }
    std::optional<ComplexMatrix> get_matrix(double param) const override {
        double angle = _pcoef * param;
        ComplexMatrix mat(2, 2);
        mat << std::cos(angle / 2), -1i * std::sin(angle / 2), -1i * std::sin(angle / 2),
            std::cos(angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        rx_gate(_target, _pcoef * param, state_vector);
    }
};

class PRYGateImpl : public POneQubitGateBase {
public:
    PRYGateImpl(UINT target, double pcoef) : POneQubitGateBase(target, pcoef){};

    PGate copy() const override { return std::make_shared<PRYGateImpl>(*this); }
    PGate get_inverse() const override { return std::make_shared<PRYGateImpl>(_target, -_pcoef); }
    std::optional<ComplexMatrix> get_matrix(double param) const override {
        double angle = _pcoef * param;
        ComplexMatrix mat(2, 2);
        mat << std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        ry_gate(_target, _pcoef * param, state_vector);
    }
};

class PRZGateImpl : public POneQubitGateBase {
public:
    PRZGateImpl(UINT target, double pcoef) : POneQubitGateBase(target, pcoef){};

    PGate copy() const override { return std::make_shared<PRZGateImpl>(*this); }
    PGate get_inverse() const override { return std::make_shared<PRZGateImpl>(_target, -_pcoef); }
    std::optional<ComplexMatrix> get_matrix(double param) const override {
        double angle = param * _pcoef;
        ComplexMatrix mat(2, 2);
        mat << std::exp(-0.5i * angle), 0, 0, std::exp(0.5i * angle);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        rz_gate(this->_target, _pcoef * param, state_vector);
    }
};

}  // namespace internal

using PRXGate = internal::PGatePtr<internal::PRXGateImpl>;
using PRYGate = internal::PGatePtr<internal::PRYGateImpl>;
using PRZGate = internal::PGatePtr<internal::PRZGateImpl>;

}  // namespace scaluq
