#pragma once

#include "../operator/pauli_operator.hpp"
#include "param_gate.hpp"
#include "update_ops.hpp"

namespace scaluq {

namespace internal {

template <std::floating_point Fp>
class ParamRXGateImpl : public ParamGateBase<Fp> {
public:
    using ParamGateBase<Fp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp>> get_inverse() const override {
        return std::make_shared<const ParamRXGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    internal::ComplexMatrix get_matrix(Fp param) const override {
        Fp angle = this->_pcoef * param;
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(angle / 2), -1i * std::sin(angle / 2), -1i * std::sin(angle / 2),
            std::cos(angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        rx_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: ParamRX\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point Fp>
class ParamRYGateImpl : public ParamGateBase<Fp> {
public:
    using ParamGateBase<Fp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    internal::ComplexMatrix get_matrix(Fp param) const override {
        Fp angle = this->_pcoef * param;
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        ry_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: ParamRY\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point Fp>
class ParamRZGateImpl : public ParamGateBase<Fp> {
public:
    using ParamGateBase<Fp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    internal::ComplexMatrix get_matrix(Fp param) const override {
        Fp angle = param * this->_pcoef;
        internal::ComplexMatrix mat(2, 2);
        mat << std::exp(-0.5i * angle), 0, 0, std::exp(0.5i * angle);
        return mat;
    }

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        rz_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: ParamRZ\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

}  // namespace internal

template <std::floating_point Fp>
using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl<Fp>>;
template <std::floating_point Fp>
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl<Fp>>;
template <std::floating_point Fp>
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_param_gate_standard_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamRXGate,
        "Specific class of parametric X rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}X}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRYGate,
        "Specific class of parametric Y rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Y}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRZGate,
        "Specific class of parametric Z rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Z}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
