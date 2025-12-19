#pragma once

#include <variant>

#include "../util/random.hpp"
#include "gate_probabilistic.hpp"
#include "param_gate_pauli.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec>
class ParamProbabilisticGateImpl : public ParamGateBase<Prec> {
    using EitherGate = std::variant<Gate<Prec>, ParamGate<Prec>>;
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<EitherGate> _gate_list;

public:
    ParamProbabilisticGateImpl(
        const std::vector<double>& distribution,
        const std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>>& gate_list);
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

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override;
    ComplexMatrix get_matrix(double) const override {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::get_matrix(): This function must not be used in "
            "ParamProbabilisticGateImpl.");
    }

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                              std::vector<double> params) const override;
    void update_quantum_state(StateVector<Prec, ExecutionSpace::HostSerialSpace>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::HostSerialSpace>& states,
                              std::vector<double> params) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                              std::vector<double> params) const override;
#endif  // SCALUQ_USE_CUDA

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

template <Precision Prec>
using ParamProbabilisticGate = internal::ParamGatePtr<internal::ParamProbabilisticGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_param_gate_probabilistic_hpp(nb::module_& m,
                                            nb::class_<ParamGate<Prec>>& param_gate_base_def) {
    bind_specific_param_gate<ParamProbabilisticGate<Prec>, Prec>(
        m,
        param_gate_base_def,
        "ParamProbabilisticGate",
        "Specific class of parametric probabilistic gate. The gate to apply is picked from a "
        "certain distribution.")
        .def(
            "gate_list",
            [](const ParamProbabilisticGate<Prec>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ParamProbabilisticGate<Prec>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
