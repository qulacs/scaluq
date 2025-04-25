#pragma once

#include "../operator/pauli_operator.hpp"
#include "param_gate.hpp"

namespace scaluq {

namespace internal {

template <Precision Prec, ExecutionSpace Space>
class ParamRXGateImpl : public ParamGateBase<Prec, Space> {
public:
    using ParamGateBase<Prec, Space>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const ParamRXGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              std::vector<double> params) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class ParamRYGateImpl : public ParamGateBase<Prec, Space> {
public:
    using ParamGateBase<Prec, Space>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              std::vector<double> params) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class ParamRZGateImpl : public ParamGateBase<Prec, Space> {
public:
    using ParamGateBase<Prec, Space>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              std::vector<double> params) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"param_coef", this->param_coef()}};
    }
};

}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl<Prec, Space>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_param_gate_standard_hpp(nb::module_& m,
                                       nb::class_<ParamGate<Prec, Space>>& param_gate_base_def) {
    DEF_PARAM_GATE(
        ParamRXGate,
        Prec,
        Space,
        "Specific class of parametric X rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{\\theta}}{2}X}$. `theta` is given as `param * param_coef`.",
        param_gate_base_def);
    DEF_PARAM_GATE(
        ParamRYGate,
        Prec,
        Space,
        "Specific class of parametric Y rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{\\theta}}{2}Y}$. `theta` is given as `param * param_coef`.",
        param_gate_base_def);
    DEF_PARAM_GATE(
        ParamRZGate,
        Prec,
        Space,
        "Specific class of parametric Z rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{\\theta}}{2}Z}$. `theta` is given as `param * param_coef`.",
        param_gate_base_def);
}
}  // namespace internal
#endif
}  // namespace scaluq
