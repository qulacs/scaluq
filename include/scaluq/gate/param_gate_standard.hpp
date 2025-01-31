#pragma once

#include "../operator/pauli_operator.hpp"
#include "param_gate.hpp"

namespace scaluq {

namespace internal {

<<<<<<< HEAD
template <Precision Prec>
class ParamRXGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRXGateImpl<Prec>>(
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class ParamRXGateImpl : public ParamGateBase<Fp, Sp> {
public:
    using ParamGateBase<Fp, Sp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const ParamRXGateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states,
                              std::vector<double> params) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector, Fp param) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                              std::vector<Fp> params) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class ParamRYGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Prec>>(
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class ParamRYGateImpl : public ParamGateBase<Fp, Sp> {
public:
    using ParamGateBase<Fp, Sp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const ParamRYGateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states,
                              std::vector<double> params) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector, Fp param) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                              std::vector<Fp> params) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class ParamRZGateImpl : public ParamGateBase<Prec> {
public:
    using ParamGateBase<Prec>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Prec>>(
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class ParamRZGateImpl : public ParamGateBase<Fp, Sp> {
public:
    using ParamGateBase<Fp, Sp>::ParamGateBase;

    std::shared_ptr<const ParamGateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const ParamRZGateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states,
                              std::vector<double> params) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector, Fp param) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                              std::vector<Fp> params) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamRZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"param_coef", this->param_coef()}};
    }
};

}  // namespace internal

<<<<<<< HEAD
template <Precision Prec>
using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl<Prec>>;
template <Precision Prec>
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl<Prec>>;
template <Precision Prec>
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl<Prec>>;

namespace internal {

#define DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(Impl, Prec)                     \
    template <>                                                                          \
    inline std::shared_ptr<const Impl<Prec>> get_from_json(const Json& j) {              \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                 \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();               \
        auto param_coef = j.at("param_coef").get<double>();                              \
        return std::make_shared<const Impl<Prec>>(vector_to_mask(targets),               \
                                                  vector_to_mask(controls),              \
                                                  static_cast<Float<Prec>>(param_coef)); \
=======
template <std::floating_point Fp, ExecutionSpace Sp>
using ParamRXGate = internal::ParamGatePtr<internal::ParamRXGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using ParamRYGate = internal::ParamGatePtr<internal::ParamRYGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using ParamRZGate = internal::ParamGatePtr<internal::ParamRZGateImpl<Fp, Sp>>;

namespace internal {

/*#define DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_TYPE(Impl, Type)             \
    template <>                                                             \
    inline std::shared_ptr<const Impl<Type>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();    \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();  \
        auto param_coef = j.at("param_coef").get<Type>();                   \
        return std::make_shared<const Impl<Type>>(                          \
            vector_to_mask(targets), vector_to_mask(controls), param_coef); \
>>>>>>> set-space
    }

#define DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_PRECISION(Prec)         \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(ParamRXGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(ParamRYGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_PRECISION(ParamRZGateImpl, Prec)

<<<<<<< HEAD
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
=======
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_PARAM_RGATE_WITH_TYPE
#undef DECLARE_GET_FROM_JSON_EACH_PARAM_RGATE_WITH_TYPE*/
>>>>>>> set-space

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
<<<<<<< HEAD
template <Precision Prec>
=======
template <std::floating_point Fp, ExecutionSpace Sp>
>>>>>>> set-space
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
