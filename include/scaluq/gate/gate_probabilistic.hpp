#pragma once

#include "../util/random.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec, ExecutionSpace Space>
class ProbabilisticGateImpl : public GateBase<Prec, Space> {
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<Gate<Prec, Space>> _gate_list;

public:
    ProbabilisticGateImpl(const std::vector<double>& distribution,
                          const std::vector<Gate<Prec, Space>>& gate_list);
    const std::vector<Gate<Prec, Space>>& gate_list() const { return _gate_list; }
    const std::vector<double>& distribution() const { return _distribution; }

    std::vector<std::uint64_t> target_qubit_list() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::target_qubit_list(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }
    std::vector<std::uint64_t> control_qubit_list() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::control_qubit_list(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }
    std::vector<std::uint64_t> control_value_list() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::control_value_list(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }
    std::vector<std::uint64_t> operand_qubit_list() const override {
        return mask_to_vector(operand_qubit_mask());
    }
    std::uint64_t target_qubit_mask() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::target_qubit_mask(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }
    std::uint64_t control_qubit_mask() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::control_qubit_mask(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }
    std::uint64_t control_value_mask() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::control_value_mask(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }
    std::uint64_t operand_qubit_mask() const override {
        std::uint64_t ret = 0ULL;
        for (const Gate<Prec, Space>& gate : _gate_list) ret |= gate->operand_qubit_mask();
        return ret;
    }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override;
    ComplexMatrix get_matrix() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::get_matrix(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Probabilistic"},
                 {"gate_list", this->gate_list()},
                 {"distribution", this->distribution()}};
    }
};
}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using ProbabilisticGate = internal::GatePtr<internal::ProbabilisticGateImpl<Prec, Space>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_probabilistic(nb::module_& m, nb::class_<Gate<Prec, Space>>& gate_base_def) {
    DEF_GATE(ProbabilisticGate,
             Prec,
             Space,
             "Specific class of probabilistic gate. The gate to apply is picked from a certain "
             "distribution.",
             gate_base_def)
        .def(
            "gate_list",
            [](const ProbabilisticGate<Prec, Space>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbabilisticGate<Prec, Space>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
