#pragma once

#include "../util/random.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp>
class ProbablisticGateImpl : public GateBase<Fp> {
    std::vector<Fp> _distribution;
    std::vector<Fp> _cumlative_distribution;
    std::vector<Gate<Fp>> _gate_list;

public:
    ProbablisticGateImpl(const std::vector<Fp>& distribution,
                         const std::vector<Gate<Fp>>& gate_list);
    const std::vector<Gate<Fp>>& gate_list() const { return _gate_list; }
    const std::vector<Fp>& distribution() const { return _distribution; }

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
    std::uint64_t operand_qubit_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::operand_qubit_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override;
    internal::ComplexMatrix<Fp> get_matrix() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};
}  // namespace internal

template <std::floating_point Fp>
using ProbablisticGate = internal::GatePtr<internal::ProbablisticGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_gate_probablistic(nb::module_& m) {
    DEF_GATE(ProbablisticGate<double>,
             "Specific class of probablistic gate. The gate to apply is picked from a cirtain "
             "distribution.")
        .def(
            "gate_list",
            [](const ProbablisticGate<double>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbablisticGate<double>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
