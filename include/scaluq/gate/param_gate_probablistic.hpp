#pragma once

#include <variant>

#include "../util/random.hpp"
#include "gate_probablistic.hpp"
#include "param_gate_pauli.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec, ExecutionSpace Space>
class ParamProbablisticGateImpl : public ParamGateBase<Prec, Space> {
    using EitherGate = std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>;
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<EitherGate> _gate_list;

public:
    ParamProbablisticGateImpl(
        const std::vector<double>& distribution,
        const std::vector<std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>>& gate_list);
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

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override;
    ComplexMatrix get_matrix(double) const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }

    void update_quantum_state(StateVector<Prec, Space>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              std::vector<double> params) const override;

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

template <Precision Prec, ExecutionSpace Space>
using ParamProbablisticGate =
    internal::ParamGatePtr<internal::ParamProbablisticGateImpl<Prec, Space>>;

namespace internal {
#define DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Prec,  \
                                                                                       Space) \
    template <>                                                                               \
    inline std::shared_ptr<const ParamProbablisticGateImpl<Prec, Space>> get_from_json(       \
        const Json& j) {                                                                      \
        auto distribution = j.at("distribution").get<std::vector<double>>();                  \
        std::vector<std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>> gate_list;       \
        const Json& tmp_list = j.at("gate_list");                                             \
        for (const Json& tmp_j : tmp_list) {                                                  \
            if (tmp_j.at("type").get<std::string>().starts_with("Param"))                     \
                gate_list.emplace_back(tmp_j.get<ParamGate<Prec, Space>>());                  \
            else                                                                              \
                gate_list.emplace_back(tmp_j.get<Gate<Prec, Space>>());                       \
        }                                                                                     \
        return std::make_shared<const ParamProbablisticGateImpl<Prec, Space>>(distribution,   \
                                                                              gate_list);     \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                               DefaultSpace)
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                               HostSpace)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                               DefaultSpace)
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                               HostSpace)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                               DefaultSpace)
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                               HostSpace)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                               DefaultSpace)
DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                               HostSpace)
#endif
#undef DECLARE_GET_FROM_JSON_PARAMPROBABLISTICGATE_WITH_PRECISION_AND_EXECUTION_SPACE
}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_param_gate_probablistic_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamProbablisticGate,
        Prec,
        Space,
        "Specific class of parametric probablistic gate. The gate to apply is picked from a "
        "cirtain "
        "distribution.")
        .def(
            "gate_list",
            [](const ParamProbablisticGate<Prec, Space>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ParamProbablisticGate<Prec, Space>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
