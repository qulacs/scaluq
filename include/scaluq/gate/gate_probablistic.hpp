#pragma once

#include "../util/random.hpp"
#include "gate_standard.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec, ExecutionSpace Space>
class ProbablisticGateImpl : public GateBase<Prec, Space> {
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<Gate<Prec, Space>> _gate_list;

public:
    ProbablisticGateImpl(const std::vector<double>& distribution,
                         const std::vector<Gate<Prec, Space>>& gate_list);
    const std::vector<Gate<Prec, Space>>& gate_list() const { return _gate_list; }
    const std::vector<double>& distribution() const { return _distribution; }

    std::vector<std::uint64_t> target_qubit_list() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::target_qubit_list(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::vector<std::uint64_t> control_qubit_list() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::control_qubit_list(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::vector<std::uint64_t> control_value_list() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::control_value_list(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::vector<std::uint64_t> operand_qubit_list() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::operand_qubit_list(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::uint64_t target_qubit_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::target_qubit_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::uint64_t control_qubit_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::control_qubit_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::uint64_t control_value_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::control_value_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::uint64_t operand_qubit_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::operand_qubit_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override;
    ComplexMatrix get_matrix() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Probablistic"},
                 {"gate_list", this->gate_list()},
                 {"distribution", this->distribution()}};
    }
};
}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using ProbablisticGate = internal::GatePtr<internal::ProbablisticGateImpl<Prec, Space>>;

namespace internal {

#define DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Prec, Space)             \
    template <>                                                                                    \
    inline std::shared_ptr<const ProbablisticGateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto distribution = j.at("distribution").get<std::vector<double>>();                       \
        auto gate_list = j.at("gate_list").get<std::vector<Gate<Prec, Space>>>();                  \
        return std::make_shared<const ProbablisticGateImpl<Prec, Space>>(distribution, gate_list); \
    }

#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                  ExecutionSpace::Host)
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                  ExecutionSpace::Default)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                  ExecutionSpace::Host)
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                  ExecutionSpace::Default)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                  ExecutionSpace::Host)
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                  ExecutionSpace::Default)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                  ExecutionSpace::Host)
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                  ExecutionSpace::Default)
#endif
#undef DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION_AND_EXECUTION_SPACE

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_probablistic(nb::module_& m, nb::class_<Gate<Prec, Space>>& gate_base_def) {
    DEF_GATE(ProbablisticGate,
             Prec,
             Space,
             "Specific class of probablistic gate. The gate to apply is picked from a cirtain "
             "distribution.",
             gate_base_def)
        .def(
            "gate_list",
            [](const ProbablisticGate<Prec, Space>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbablisticGate<Prec, Space>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
