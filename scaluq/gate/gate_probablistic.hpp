#pragma once

#include "../util/random.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
class ProbablisticGateImpl : public GateBase {
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<Gate> _gate_list;

public:
    ProbablisticGateImpl(const std::vector<double>& distribution,
                         const std::vector<Gate>& gate_list)
        : GateBase(0, 0), _distribution(distribution), _gate_list(gate_list) {
        std::uint64_t n = distribution.size();
        if (n == 0) {
            throw std::runtime_error("At least one gate is required.");
        }
        if (n != gate_list.size()) {
            throw std::runtime_error("distribution and gate_list have different size.");
        }
        _cumulative_distribution.resize(n + 1);
        std::partial_sum(
            distribution.begin(), distribution.end(), _cumulative_distribution.begin() + 1);
        if (std::abs(_cumulative_distribution.back() - 1.) > 1e-6) {
            throw std::runtime_error("Sum of distribution must be equal to 1.");
        }
    }
    const std::vector<Gate>& gate_list() const { return _gate_list; }
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

    Gate get_inverse() const override {
        std::vector<Gate> inv_gate_list;
        inv_gate_list.reserve(_gate_list.size());
        std::ranges::transform(_gate_list, std::back_inserter(inv_gate_list), [](const Gate& gate) {
            return gate->get_inverse();
        });
        return std::make_shared<const ProbablisticGateImpl>(_distribution, inv_gate_list);
    }
    internal::ComplexMatrix get_matrix() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

    void update_quantum_state(StateVector& state_vector) const override {
        Random random;
        double r = random.uniform();
        std::uint64_t i = std::distance(_cumulative_distribution.begin(),
                                        std::ranges::upper_bound(_cumulative_distribution, r)) -
                          1;
        if (i >= _gate_list.size()) i = _gate_list.size() - 1;
        _gate_list[i]->update_quantum_state(state_vector);
    }

    // これでいいのか？わからない...
    void update_quantum_state(StateVectorBatched& states) const override {
        std::vector<std::uint64_t> indices(states.batch_size());
        std::vector<double> r(states.batch_size());

        Random random;
        for (std::size_t i = 0; i < states.batch_size(); ++i) {
            r[i] = random.uniform();
            indices[i] = std::distance(_cumulative_distribution.begin(),
                                       std::ranges::upper_bound(_cumulative_distribution, r[i])) -
                         1;
            if (indices[i] >= _gate_list.size()) indices[i] = _gate_list.size() - 1;
            auto state_vector = states.get_state_vector_at(i);
            _gate_list[indices[i]]->update_quantum_state(state_vector);
            Kokkos::parallel_for(
                "update_states", states.dim(), KOKKOS_CLASS_LAMBDA(const int j) {
                    states._raw(i, j) = state_vector._raw(j);
                });
        }
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        const auto dist = distribution();
        ss << indent << "Gate Type: Probablistic\n";
        for (std::size_t i = 0; i < dist.size(); ++i) {
            ss << indent << "  --------------------\n";
            ss << indent << "  Probability: " << dist[i] << "\n";
            ss << gate_list()[i]->to_string(indent + "  ") << (i == dist.size() - 1 ? "" : "\n");
        }
        return ss.str();
    }
};
}  // namespace internal

using ProbablisticGate = internal::GatePtr<internal::ProbablisticGateImpl>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_gate_probablistic(nb::module_& m) {
    DEF_GATE(ProbablisticGate,
             "Specific class of probablistic gate. The gate to apply is picked from a cirtain "
             "distribution.")
        .def(
            "gate_list",
            [](const ProbablisticGate& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbablisticGate& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
