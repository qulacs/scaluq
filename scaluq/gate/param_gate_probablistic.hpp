#pragma once

#include <variant>

#include "../util/random.hpp"
#include "gate.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
class PProbablisticGateImpl : public ParamGateBase {
    using EitherGate = std::variant<Gate, ParamGate>;
    std::vector<double> _distribution;
    std::vector<double> _cumlative_distribution;
    std::vector<EitherGate> _gate_list;

public:
    PProbablisticGateImpl(const std::vector<double>& distribution,
                          const std::vector<std::variant<Gate, ParamGate>>& gate_list)
        : ParamGateBase(  // make OR(target mask) and OR(control mask) at first
              [&gate_list] {
                  UINT mask_sum = 0;
                  for (const auto& gate : gate_list) {
                      mask_sum |= std::visit(
                          [](const auto& g) { return g->get_target_qubit_mask(); }, gate);
                  }
                  return mask_sum;
              }(),
              [&gate_list] {
                  UINT mask_sum = 0;
                  for (const auto& gate : gate_list) {
                      mask_sum |= std::visit(
                          [](const auto& g) { return g->get_control_qubit_mask(); }, gate);
                  }
                  return mask_sum;
              }()),
          _distribution(distribution),
          _gate_list(gate_list) {
        UINT n = distribution.size();
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

    ParamGate get_inverse() const override {
        std::vector<EitherGate> inv_gate_list;
        inv_gate_list.reserve(_gate_list.size());
        std::ranges::transform(
            _gate_list, std::back_inserter(inv_gate_list), [](const EitherGate& gate) {
                return std::visit([](const auto& g) { return EitherGate{g->get_inverse()}; }, gate);
            });
        return std::make_shared<const PProbablisticGateImpl>(_distribution, inv_gate_list);
    }
    std::optional<ComplexMatrix> get_matrix(double param) const override {
        if (_gate_list.size() == 1) {
            const auto& gate = _gate_list[0];
            if (gate.index() == 0) {
                return std::get<0>(gate)->get_matrix();
            } else {
                return std::get<1>(gate)->get_matrix(param);
            }
        }
        return std::nullopt;
    }

    void update_quantum_state(StateVector& state_vector, double param) const override {
        Random random;
        double r = random.uniform();
        UINT i = std::distance(_cumlative_distribution.begin(),
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

using PProbablisticGate = internal::ParamGatePtr<internal::PProbablisticGateImpl>;
}  // namespace scaluq
