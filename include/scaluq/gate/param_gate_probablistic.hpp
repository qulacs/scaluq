#pragma once

#include <variant>

#include "../util/random.hpp"
#include "gate.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
template <std::floating_point Fp>
class ParamProbablisticGateImpl : public ParamGateBase<Fp> {
    using EitherGate = std::variant<Gate<Fp>, ParamGate<Fp>>;
    std::vector<Fp> _distribution;
    std::vector<Fp> _cumlative_distribution;
    std::vector<EitherGate> _gate_list;

public:
    ParamProbablisticGateImpl(const std::vector<Fp>& distribution,
                              const std::vector<std::variant<Gate<Fp>, ParamGate<Fp>>>& gate_list);
    const std::vector<EitherGate>& gate_list() const { return _gate_list; }
    const std::vector<Fp>& distribution() const { return _distribution; }

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

    std::shared_ptr<const ParamGateBase<Fp>> get_inverse() const override;
    internal::ComplexMatrix<Fp> get_matrix(Fp) const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }

    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override;

    std::string to_string(const std::string& indent) const override;
};
}  // namespace internal

template <std::floating_point Fp>
using ParamProbablisticGate = internal::ParamGatePtr<internal::ParamProbablisticGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_param_gate_probablistic_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamProbablisticGate<double>,
        "Specific class of parametric probablistic gate. The gate to apply is picked from a "
        "cirtain "
        "distribution.")
        .def(
            "gate_list",
            [](const ParamProbablisticGate<double>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ParamProbablisticGate<double>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
