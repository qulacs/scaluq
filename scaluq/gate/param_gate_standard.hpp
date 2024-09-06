#pragma once

#include "../operator/pauli_operator.hpp"
#include "param_gate.hpp"
#include "update_ops.hpp"

namespace scaluq {

namespace internal {

class ParamRXGateImpl : public ParamGateBase {
public:
    using ParamGateBase::ParamGateBase;

    ParamGate get_inverse() const override {
        return std::make_shared<const ParamRXGateImpl>(_target_mask, _control_mask, -_pcoef);
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

class ParamRYGateImpl : public ParamGateBase {
public:
    using ParamGateBase::ParamGateBase;

    ParamGate get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl>(_target_mask, _control_mask, -_pcoef);
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

class ParamRZGateImpl : public ParamGateBase {
public:
    using ParamGateBase::ParamGateBase;

    ParamGate get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl>(_target_mask, _control_mask, -_pcoef);
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

using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl>;
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl>;
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_param_gate_standard_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamPauliRotationGate,
        "Specific class of parametric multi-qubit pauli-rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}P}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
