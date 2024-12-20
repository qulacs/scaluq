#pragma once

#include "../operator/pauli_operator.hpp"
#include "param_gate.hpp"

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
    internal::ComplexMatrix<Fp> get_matrix(Fp param) const override;

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class ParamRYGateImpl : public ParamGateBase<Fp> {
public:
    using ParamGateBase<Fp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    internal::ComplexMatrix<Fp> get_matrix(Fp param) const override;

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class ParamRZGateImpl : public ParamGateBase<Fp> {
public:
    using ParamGateBase<Fp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    internal::ComplexMatrix<Fp> get_matrix(Fp param) const override;

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override;

    std::string to_string(const std::string& indent) const override;
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
template <std::floating_point Fp>
void bind_gate_param_gate_standard_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamRXGate,
        Fp,
        "Specific class of parametric X rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}X}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRYGate,
        Fp,
        "Specific class of parametric Y rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Y}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRZGate,
        Fp,
        "Specific class of parametric Z rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Z}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
