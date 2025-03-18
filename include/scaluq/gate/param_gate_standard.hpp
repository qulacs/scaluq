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
            this->_target_mask, this->_control_mask, -this->_pcoef);
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
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class ParamRYGateImpl : public ParamGateBase<Prec, Space> {
public:
    using ParamGateBase<Prec, Space>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
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
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class ParamRZGateImpl : public ParamGateBase<Prec, Space> {
public:
    using ParamGateBase<Prec, Space>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
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

namespace internal {

#define DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Impl, Prec, Space) \
    template <>                                                                                 \
    inline std::shared_ptr<const Impl<Prec, Space>> get_from_json(const Json& j) {              \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                        \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                      \
        auto param_coef = j.at("param_coef").get<double>();                                     \
        return std::make_shared<const Impl<Prec, Space>>(vector_to_mask(targets),               \
                                                         vector_to_mask(controls),              \
                                                         static_cast<Float<Prec>>(param_coef)); \
    }

#define DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Prec, Space) \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                      \
        ParamRXGateImpl, Prec, Space)                                                          \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                      \
        ParamRYGateImpl, Prec, Space)                                                          \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                      \
        ParamRZGateImpl, Prec, Space)

#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                          ExecutionSpace::Default)
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                          ExecutionSpace::Host)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                          ExecutionSpace::Default)
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                          ExecutionSpace::Host)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                          ExecutionSpace::Default)
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                          ExecutionSpace::Host)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                          ExecutionSpace::Default)
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                          ExecutionSpace::Host)
#endif
#undef DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE
#undef DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE

}  // namespace internal

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
