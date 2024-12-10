#include <scaluq/gate/gate_probablistic.hpp>

#include "../util/template.hpp"

namespace scaluq::internal {
FLOAT(Fp)
ProbablisticGateImpl<Fp>::ProbablisticGateImpl(const std::vector<Fp>& distribution,
                                               const std::vector<Gate<Fp>>& gate_list)
    : GateBase<Fp>(0, 0), _distribution(distribution), _gate_list(gate_list) {
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
FLOAT(Fp)
std::shared_ptr<const GateBase<Fp>> ProbablisticGateImpl<Fp>::get_inverse() const {
    std::vector<Gate<Fp>> inv_gate_list;
    inv_gate_list.reserve(_gate_list.size());
    std::ranges::transform(_gate_list, std::back_inserter(inv_gate_list), [](const Gate<Fp>& gate) {
        return gate->get_inverse();
    });
    return std::make_shared<const ProbablisticGateImpl>(_distribution, inv_gate_list);
}
FLOAT(Fp)
void ProbablisticGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    Random random;
    Fp r = random.uniform();
    std::uint64_t i = std::distance(_cumulative_distribution.begin(),
                                    std::ranges::upper_bound(_cumulative_distribution, r)) -
                      1;
    if (i >= _gate_list.size()) i = _gate_list.size() - 1;
    _gate_list[i]->update_quantum_state(state_vector);
}
FLOAT(Fp)
void ProbablisticGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    std::vector<std::uint64_t> indices(states.batch_size());
    std::vector<Fp> r(states.batch_size());

    Random random;
    for (std::size_t i = 0; i < states.batch_size(); ++i) {
        r[i] = random.uniform();
        indices[i] = std::distance(_cumulative_distribution.begin(),
                                   std::ranges::upper_bound(_cumulative_distribution, r[i])) -
                     1;
        if (indices[i] >= _gate_list.size()) indices[i] = _gate_list.size() - 1;
        auto state_vector = StateVector<Fp>(Kokkos::subview(states._raw, i, Kokkos::ALL));
        _gate_list[indices[i]]->update_quantum_state(state_vector);
        Kokkos::parallel_for(
            "update_states", states.dim(), KOKKOS_CLASS_LAMBDA(const int j) {
                states._raw(i, j) = state_vector._raw(j);
            });
    }
}
FLOAT(Fp)
std::string ProbablisticGateImpl<Fp>::to_string(const std::string& indent) const {
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
FLOAT_DECLARE_CLASS(ProbablisticGateImpl)
}  // namespace scaluq::internal
