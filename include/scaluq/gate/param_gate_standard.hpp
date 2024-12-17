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

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
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
    internal::ComplexMatrix<Fp> get_matrix(Fp param) const override;

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
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
    internal::ComplexMatrix<Fp> get_matrix(Fp param) const override;

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
    }
};

}  // namespace internal

template <std::floating_point Fp>
using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl<Fp>>;
template <std::floating_point Fp>
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl<Fp>>;
template <std::floating_point Fp>
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl<Fp>>;

namespace internal {

#define DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_TYPE(Type)                              \
    template <>                                                                        \
    inline std::shared_ptr<const ParamRXGateImpl<Type>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();               \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();             \
        auto param_coef = j.at("param_coef").get<Type>();                              \
        return std::make_shared<const ParamRXGateImpl<Type>>(                          \
            vector_to_mask(targets), vector_to_mask(controls), param_coef);            \
    }                                                                                  \
    template <>                                                                        \
    inline std::shared_ptr<const ParamRYGateImpl<Type>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();               \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();             \
        auto param_coef = j.at("param_coef").get<Type>();                              \
        return std::make_shared<const ParamRYGateImpl<Type>>(                          \
            vector_to_mask(targets), vector_to_mask(controls), param_coef);            \
    }                                                                                  \
    template <>                                                                        \
    inline std::shared_ptr<const ParamRZGateImpl<Type>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();               \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();             \
        auto param_coef = j.at("param_coef").get<Type>();                              \
        return std::make_shared<const ParamRZGateImpl<Type>>(                          \
            vector_to_mask(targets), vector_to_mask(controls), param_coef);            \
    }

DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_TYPE(float)

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_param_gate_standard_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamRXGate,
        double,
        "Specific class of parametric X rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}X}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRYGate,
        double,
        "Specific class of parametric Y rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Y}$. `angle` is given as `param * param_coef`.");
    DEF_PARAM_GATE(
        ParamRZGate,
        double,
        "Specific class of parametric Z rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}Z}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
