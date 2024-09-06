#pragma once

#include <variant>

#include "../util/random.hpp"
#include "gate.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
class ParamProbablisticGateImpl : public ParamGateBase {
    using EitherGate = std::variant<Gate, ParamGate>;
    std::vector<double> _distribution;
    std::vector<double> _cumlative_distribution;
    std::vector<EitherGate> _gate_list;

public:
    ParamProbablisticGateImpl(const std::vector<double>& distribution,
                              const std::vector<std::variant<Gate, ParamGate>>& gate_list)
        : ParamGateBase(0, 0), _distribution(distribution), _gate_list(gate_list) {
        std::uint64_t n = distribution.size();
        if (n == 0) {
            throw std::runtime_error("At least one gate is required.");
        }
        if (n != gate_list.size()) {
            throw std::runtime_error("distribution and gate_list have different size.");
        }
        _cumlative_distribution.resize(n + 1);
        std::partial_sum(
            distribution.begin(), distribution.end(), _cumlative_distribution.begin() + 1);
        if (std::abs(_cumlative_distribution.back() - 1.) > 1e-6) {
            throw std::runtime_error("Sum of distribution must be equal to 1.");
        }
    }
    const std::vector<std::variant<Gate, ParamGate>>& gate_list() const { return _gate_list; }
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

    ParamGate get_inverse() const override {
        std::vector<EitherGate> inv_gate_list;
        inv_gate_list.reserve(_gate_list.size());
        std::ranges::transform(
            _gate_list, std::back_inserter(inv_gate_list), [](const EitherGate& gate) {
                return std::visit([](const auto& g) { return EitherGate{g->get_inverse()}; }, gate);
            });
        return std::make_shared<const ParamProbablisticGateImpl>(_distribution, inv_gate_list);
    }
    ComplexMatrix get_matrix(double) const override {
        throw std::runtime_error(
            "ParamProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ParamProbablisticGateImpl.");
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        Random random;
        double r = random.uniform();
        std::uint64_t i = std::distance(_cumlative_distribution.begin(),
                                        std::ranges::upper_bound(_cumlative_distribution, r)) -
                          1;
        if (i >= _gate_list.size()) i = _gate_list.size() - 1;
        const auto& gate = _gate_list[i];
        if (gate.index() == 0) {
            std::get<0>(gate)->update_quantum_state(state_vector);
        } else {
            std::get<1>(gate)->update_quantum_state(state_vector, param);
        }
    }
};
}  // namespace internal

using ParamProbablisticGate = internal::ParamGatePtr<internal::ParamProbablisticGateImpl>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_param_gate_probablistic_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamProbablisticGate,
        "Specific class of parametric probablistic gate. The gate to apply is picked from a "
        "cirtain "
        "distribution.")
        .def(
            "gate_list",
            [](const ParamProbablisticGate& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ParamProbablisticGate& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
