#pragma once

#include "../util/random.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec>
class ProbabilisticGateImpl : public GateBase<Prec> {
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<Gate<Prec>> _gate_list;

public:
    ProbabilisticGateImpl(const std::vector<double>& distribution,
                          const std::vector<Gate<Prec>>& gate_list);
    const std::vector<Gate<Prec>>& gate_list() const { return _gate_list; }
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
        for (const Gate<Prec>& gate : _gate_list) ret |= gate->operand_qubit_mask();
        return ret;
    }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override;
    ComplexMatrix get_matrix() const override {
        throw std::runtime_error(
            "ProbabilisticGateImpl::get_matrix(): This function must not be used in "
            "ProbabilisticGateImpl.");
    }

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::HostSerialSpace>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::HostSerialSpace>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Probabilistic"},
                 {"gate_list", this->gate_list()},
                 {"distribution", this->distribution()}};
    }
};
}  // namespace internal

template <Precision Prec>
using ProbabilisticGate = internal::GatePtr<internal::ProbabilisticGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_probabilistic(nb::module_& m, nb::class_<Gate<Prec>>& gate_base_def) {
    bind_specific_gate<ProbabilisticGate<Prec>, Prec>(
        m,
        gate_base_def,
        "ProbabilisticGate",
        "Specific class of probabilistic gate. The gate to apply is picked from a certain "
        "distribution.")
        .def(
            "gate_list",
            [](const ProbabilisticGate<Prec>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbabilisticGate<Prec>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
