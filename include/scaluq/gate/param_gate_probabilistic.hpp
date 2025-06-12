#pragma once

#include <variant>

#include "../util/random.hpp"
#include "gate_probabilistic.hpp"
#include "param_gate_pauli.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec, ExecutionSpace Space>
class ParamProbabilisticGateImpl : public ParamGateBase<Prec, Space> {
    using EitherGate = std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>;
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<EitherGate> _gate_list;

public:
    ParamProbabilisticGateImpl(
        const std::vector<double>& distribution,
        const std::vector<std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>>& gate_list);
    const std::vector<EitherGate>& gate_list() const { return _gate_list; }
    const std::vector<double>& distribution() const { return _distribution; }

    std::vector<std::uint64_t> target_qubit_list() const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::target_qubit_list(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }
    std::vector<std::uint64_t> control_qubit_list() const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::control_qubit_list(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }
    std::vector<std::uint64_t> control_value_list() const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::control_value_list(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }
    std::vector<std::uint64_t> operand_qubit_list() const override {
        return mask_to_vector(operand_qubit_mask());
    }
    std::uint64_t target_qubit_mask() const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::target_qubit_mask(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }
    std::uint64_t control_qubit_mask() const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::control_qubit_mask(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }
    std::uint64_t control_value_mask() const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::control_value_mask(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }
    std::uint64_t operand_qubit_mask() const override {
        std::uint64_t ret = 0ULL;
        for (const EitherGate& gate : _gate_list)
            ret |= std::visit([&](const auto& gate) { return gate->operand_qubit_mask(); }, gate);
        return ret;
    }

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override;
    ComplexMatrix get_matrix(double) const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::get_matrix(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }

    void update_quantum_state(StateVector<Prec, Space>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              std::vector<double> params) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamProbabilistic"},
                 {"gate_list", Json::array()},
                 {"distribution", this->distribution()}};

        for (const auto& gate : this->gate_list()) {
            std::visit([&](auto&& arg) { j["gate_list"].push_back(arg); }, gate);
        }
    }
};
}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using ParamProbabilisticGate =
    internal::ParamGatePtr<internal::ParamProbabilisticGateImpl<Prec, Space>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_param_gate_probabilistic_hpp(
    nb::module_& m, nb::class_<ParamGate<Prec, Space>>& param_gate_base_def) {
    DEF_PARAM_GATE(
        ParamProbabilisticGate,
        Prec,
        Space,
        "Specific class of parametric probabilistic gate. The gate to apply is picked from a "
        "certain distribution.",
        param_gate_base_def)
        .def(
            "gate_list",
            [](const ParamProbabilisticGate<Prec, Space>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ParamProbabilisticGate<Prec, Space>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
