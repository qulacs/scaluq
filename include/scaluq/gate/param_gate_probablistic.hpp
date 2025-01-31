#pragma once

#include <variant>

#include "../util/random.hpp"
#include "gate_probablistic.hpp"
#include "param_gate_pauli.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
<<<<<<< HEAD
template <Precision Prec>
class ParamProbablisticGateImpl : public ParamGateBase<Prec> {
    using EitherGate = std::variant<Gate<Prec>, ParamGate<Prec>>;
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class ParamProbablisticGateImpl : public ParamGateBase<Fp, Sp> {
    using EitherGate = std::variant<Gate<Fp, Sp>, ParamGate<Fp, Sp>>;
    std::vector<Fp> _distribution;
    std::vector<Fp> _cumulative_distribution;
>>>>>>> set-space
    std::vector<EitherGate> _gate_list;

public:
    ParamProbablisticGateImpl(
<<<<<<< HEAD
        const std::vector<double>& distribution,
        const std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>>& gate_list);
=======
        const std::vector<Fp>& distribution,
        const std::vector<std::variant<Gate<Fp, Sp>, ParamGate<Fp, Sp>>>& gate_list);
>>>>>>> set-space
    const std::vector<EitherGate>& gate_list() const { return _gate_list; }
    const std::vector<double>& distribution() const { return _distribution; }

    std::vector<std::uint64_t> target_qubit_list() const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::target_qubit_list(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }
    std::vector<std::uint64_t> control_qubit_list() const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::control_qubit_list(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }
    std::vector<std::uint64_t> operand_qubit_list() const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::operand_qubit_list(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }
    std::uint64_t target_qubit_mask() const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::target_qubit_mask(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }
    std::uint64_t control_qubit_mask() const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::control_qubit_mask(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }
    std::uint64_t operand_qubit_mask() const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::operand_qubit_mask(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }

<<<<<<< HEAD
    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override;
    ComplexMatrix get_matrix(double) const override {
=======
    std::shared_ptr<const ParamGateBase<Fp, Sp>> get_inverse() const override;
    internal::ComplexMatrix<Fp> get_matrix(Fp) const override {
>>>>>>> set-space
        throw std::runtime_error(
            "ParamProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }

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
        j = Json{{"type", "ParamProbablistic"},
                 {"gate_list", Json::array()},
                 {"distribution", this->distribution()}};

        for (const auto& gate : this->gate_list()) {
            std::visit([&](auto&& arg) { j["gate_list"].push_back(arg); }, gate);
        }
    }
};
}  // namespace internal

<<<<<<< HEAD
template <Precision Prec>
using ParamProbablisticGate = internal::ParamGatePtr<internal::ParamProbablisticGateImpl<Prec>>;

namespace internal {
#define DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION(Prec)                         \
=======
template <std::floating_point Fp, ExecutionSpace Sp>
using ParamProbablisticGate = internal::ParamGatePtr<internal::ParamProbablisticGateImpl<Fp, Sp>>;

namespace internal {
/*
#define DECLARE_GET_FROM_JSON_PARAM_PROBGATE_WITH_TYPE(Type)                                     \
>>>>>>> set-space
    template <>                                                                                  \
    inline std::shared_ptr<const ParamProbablisticGateImpl<Prec>> get_from_json(const Json& j) { \
        auto distribution = j.at("distribution").get<std::vector<double>>();                     \
        std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>> gate_list;                        \
        const Json& tmp_list = j.at("gate_list");                                                \
        for (const Json& tmp_j : tmp_list) {                                                     \
            if (tmp_j.at("type").get<std::string>().starts_with("Param"))                        \
                gate_list.emplace_back(tmp_j.get<ParamGate<Prec>>());                            \
            else                                                                                 \
                gate_list.emplace_back(tmp_j.get<Gate<Prec>>());                                 \
        }                                                                                        \
        return std::make_shared<const ParamProbablisticGateImpl<Prec>>(distribution, gate_list); \
    }
<<<<<<< HEAD
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_PARAM_PROBABLISTICGATE_WITH_PRECISION
=======
DECLARE_GET_FROM_JSON_PARAM_PROBGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_PARAM_PROBGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_PARAM_PROBGATE_WITH_TYPE
*/
>>>>>>> set-space
}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_param_gate_probablistic_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamProbablisticGate,
        Prec,
        "Specific class of parametric probablistic gate. The gate to apply is picked from a "
        "cirtain "
        "distribution.")
        .def(
            "gate_list",
            [](const ParamProbablisticGate<Fp>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ParamProbablisticGate<Fp>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
