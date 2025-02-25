#include <scaluq/gate/gate_probablistic.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec, ExecutionSpace Space>
ProbablisticGateImpl<Prec, Space>::ProbablisticGateImpl(
    const std::vector<double>& distribution, const std::vector<Gate<Prec, Space>>& gate_list)
    : GateBase<Prec, Space>(0, 0), _distribution(distribution), _gate_list(gate_list) {
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
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const GateBase<Prec, Space>> ProbablisticGateImpl<Prec, Space>::get_inverse()
    const {
    std::vector<Gate<Prec, Space>> inv_gate_list;
    inv_gate_list.reserve(_gate_list.size());
    std::ranges::transform(_gate_list,
                           std::back_inserter(inv_gate_list),
                           [](const Gate<Prec, Space>& gate) { return gate->get_inverse(); });
    return std::make_shared<const ProbablisticGateImpl<Prec, Space>>(_distribution, inv_gate_list);
}
template <Precision Prec, ExecutionSpace Space>
void ProbablisticGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    Random random;
    double r = random.uniform();
    std::uint64_t i = std::distance(_cumulative_distribution.begin(),
                                    std::ranges::upper_bound(_cumulative_distribution, r)) -
                      1;
    if (i >= _gate_list.size()) i = _gate_list.size() - 1;
    _gate_list[i]->update_quantum_state(state_vector);
}
template <Precision Prec, ExecutionSpace Space>
void ProbablisticGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    std::vector<std::uint64_t> indices(states.batch_size());
    std::vector<double> r(states.batch_size());

    Random random;
    for (std::size_t i = 0; i < states.batch_size(); ++i) {
        r[i] = random.uniform();
        indices[i] = std::distance(_cumulative_distribution.begin(),
                                   std::ranges::upper_bound(_cumulative_distribution, r[i])) -
                     1;
        if (indices[i] >= _gate_list.size()) indices[i] = _gate_list.size() - 1;
        auto state_vector = StateVector<Prec, Space>(Kokkos::subview(states._raw, i, Kokkos::ALL));
        _gate_list[indices[i]]->update_quantum_state(state_vector);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Space>(0, states.dim()),
            KOKKOS_CLASS_LAMBDA(const int j) { states._raw(i, j) = state_vector._raw(j); });
    }
}
template <Precision Prec, ExecutionSpace Space>
std::string ProbablisticGateImpl<Prec, Space>::to_string(const std::string& indent) const {
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
SCALUQ_DECLARE_CLASS_FOR_PRECISION_AND_EXECUTION_SPACE(ProbablisticGateImpl)
}  // namespace scaluq::internal
