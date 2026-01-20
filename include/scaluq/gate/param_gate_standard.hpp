#pragma once

#include "../operator/pauli_operator.hpp"
#include "param_gate.hpp"

namespace scaluq {

namespace internal {

template <Precision Prec>
class ParamRXGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRXGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                              std::vector<double> params) const override;
    void update_quantum_state(StateVector<Prec, ExecutionSpace::HostSerial>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
                              std::vector<double> params) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                              std::vector<double> params) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec>
class ParamRYGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                              std::vector<double> params) const override;
    void update_quantum_state(StateVector<Prec, ExecutionSpace::HostSerial>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
                              std::vector<double> params) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                              std::vector<double> params) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec>
class ParamRZGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                              std::vector<double> params) const override;
    void update_quantum_state(StateVector<Prec, ExecutionSpace::HostSerial>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
                              std::vector<double> params) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                              std::vector<double> params) const override;
#endif  // SCALUQ_USE_CUDA

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

template <Precision Prec>
using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl<Prec>>;
template <Precision Prec>
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl<Prec>>;
template <Precision Prec>
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_param_gate_standard_hpp(nb::module_& m,
                                       nb::class_<ParamGate<Prec>>& param_gate_base_def) {
    bind_specific_param_gate<ParamRXGate<Prec>, Prec>(
        m,
        param_gate_base_def,
        "ParamRXGate",
        "Specific class of parametric X rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{\\theta}}{2}X}$. `theta` is given as `param * param_coef`.");
    bind_specific_param_gate<ParamRYGate<Prec>, Prec>(
        m,
        param_gate_base_def,
        "ParamRYGate",
        "Specific class of parametric Y rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{\\theta}}{2}Y}$. `theta` is given as `param * param_coef`.");
    bind_specific_param_gate<ParamRZGate<Prec>, Prec>(
        m,
        param_gate_base_def,
        "ParamRZGate",
        "Specific class of parametric Z rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{\\theta}}{2}Z}$. `theta` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
