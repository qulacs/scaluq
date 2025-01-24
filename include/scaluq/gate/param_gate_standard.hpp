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
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states,
                              std::vector<double> params) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec>
class ParamRYGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states,
                              std::vector<double> params) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
    }
};

template <Precision Prec>
class ParamRZGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

    void update_quantum_state(StateVector<Prec>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states,
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

template <Precision Prec>
using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl<Prec>>;
template <Precision Prec>
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl<Prec>>;
template <Precision Prec>
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl<Prec>>;

namespace internal {

#define DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(Impl, Prec)        \
    template <>                                                             \
    inline std::shared_ptr<const Impl<Prec>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();    \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();  \
        auto param_coef = j.at("param_coef").get<double>();                 \
        return std::make_shared<const Impl<Prec>>(                          \
            vector_to_mask(targets), vector_to_mask(controls), param_coef); \
    }

#define DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION(Prec)         \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(ParamRXGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(ParamRYGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(ParamRZGateImpl, Prec)

#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION
#undef DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_param_gate_standard_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamRXGate,
        Prec,
        "Specific class of parametric X rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}X}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRYGate,
        Prec,
        "Specific class of parametric Y rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Y}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRZGate,
        Prec,
        "Specific class of parametric Z rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Z}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
