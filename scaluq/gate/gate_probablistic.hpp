#pragma once

#include "../util/random.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
class ProbablisticGateImpl : public GateBase {
    std::vector<double> _distribution;
    std::vector<double> _cumlative_distribution;
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
        _cumlative_distribution.resize(n + 1);
        std::partial_sum(
            distribution.begin(), distribution.end(), _cumlative_distribution.begin() + 1);
        if (std::abs(_cumlative_distribution.back() - 1.) > 1e-6) {
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
        std::uint64_t i = std::distance(_cumlative_distribution.begin(),
                                        std::ranges::upper_bound(_cumlative_distribution, r)) -
                          1;
        if (i >= _gate_list.size()) i = _gate_list.size() - 1;
        _gate_list[i]->update_quantum_state(state_vector);
    }
};
}  // namespace internal

using ProbablisticGate = internal::GatePtr<internal::ProbablisticGateImpl>;
}  // namespace scaluq
