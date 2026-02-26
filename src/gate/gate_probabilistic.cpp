#include <scaluq/gate/gate_probabilistic.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
namespace {
template <Precision Prec>
void flatten_probabilistic_gate(double prob_prefix,
                                const Gate<Prec>& gate,
                                std::vector<double>& flattened_distribution,
                                std::vector<Gate<Prec>>& flattened_gate_list) {
    if (gate.gate_type() != GateType::Probabilistic) {
        flattened_distribution.push_back(prob_prefix);
        flattened_gate_list.push_back(gate);
        return;
    }
    auto probabilistic_gate = ProbabilisticGate<Prec>(gate);
    const auto& distribution = probabilistic_gate->distribution();
    const auto& gate_list = probabilistic_gate->gate_list();
    for (std::size_t i = 0; i < distribution.size(); ++i) {
        flatten_probabilistic_gate(
            prob_prefix * distribution[i], gate_list[i], flattened_distribution, flattened_gate_list);
    }
}
}  // namespace

template <Precision Prec>
ProbabilisticGateImpl<Prec>::ProbabilisticGateImpl(const std::vector<double>& distribution,
                                                   const std::vector<Gate<Prec>>& gate_list)
    : GateBase<Prec>(0, 0, 0) {
    std::uint64_t n = distribution.size();
    if (n == 0) {
        throw std::runtime_error("At least one gate is required.");
    }
    if (n != gate_list.size()) {
        throw std::runtime_error("distribution and gate_list have different size.");
    }

    _distribution.clear();
    _gate_list.clear();
    _distribution.reserve(n);
    _gate_list.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        flatten_probabilistic_gate(distribution[i], gate_list[i], _distribution, _gate_list);
    }

    _cumulative_distribution.resize(_distribution.size() + 1);
    std::partial_sum(
        _distribution.begin(), _distribution.end(), _cumulative_distribution.begin() + 1);
    if (std::abs(_cumulative_distribution.back() - 1.) > 1e-6) {
        throw std::runtime_error("Sum of distribution must be equal to 1.");
    }
}
template <Precision Prec>
std::shared_ptr<const GateBase<Prec>> ProbabilisticGateImpl<Prec>::get_inverse() const {
    std::vector<Gate<Prec>> inv_gate_list;
    inv_gate_list.reserve(_gate_list.size());
    std::ranges::transform(_gate_list,
                           std::back_inserter(inv_gate_list),
                           [](const Gate<Prec>& gate) { return gate->get_inverse(); });
    return std::make_shared<const ProbabilisticGateImpl<Prec>>(_distribution, inv_gate_list);
}

template <Precision Prec>
std::string ProbabilisticGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    const auto dist = distribution();
    ss << indent << "Gate Type: Probabilistic\n";
    for (std::size_t i = 0; i < dist.size(); ++i) {
        ss << indent << "  --------------------\n";
        ss << indent << "  Probability: " << dist[i] << "\n";
        ss << gate_list()[i]->to_string(indent + "  ") << (i == dist.size() - 1 ? "" : "\n");
    }
    return ss.str();
}
#define DEFINE_PROBABILISTIC_GATE_UPDATE(Space)                                                    \
    template <Precision Prec>                                                                      \
    void ProbabilisticGateImpl<Prec>::update_quantum_state(StateVector<Prec, Space>& state_vector) \
        const {                                                                                    \
        Random random;                                                                             \
        double r = random.uniform();                                                               \
        std::uint64_t i = std::distance(_cumulative_distribution.begin(),                          \
                                        std::ranges::upper_bound(_cumulative_distribution, r)) -   \
                          1;                                                                       \
        if (i >= _gate_list.size()) i = _gate_list.size() - 1;                                     \
        _gate_list[i]->update_quantum_state(state_vector);                                         \
    }                                                                                              \
    template <Precision Prec>                                                                      \
    void ProbabilisticGateImpl<Prec>::update_quantum_state(                                        \
        StateVectorBatched<Prec, Space>& states) const {                                           \
        std::vector<std::uint64_t> indices(states.batch_size());                                   \
        std::vector<double> r(states.batch_size());                                                \
                                                                                                   \
        Random random;                                                                             \
        for (std::size_t i = 0; i < states.batch_size(); ++i) {                                    \
            r[i] = random.uniform();                                                               \
            indices[i] = std::distance(_cumulative_distribution.begin(),                           \
                                       std::ranges::upper_bound(_cumulative_distribution, r[i])) - \
                         1;                                                                        \
            if (indices[i] >= _gate_list.size()) indices[i] = _gate_list.size() - 1;               \
            auto state_vector =                                                                    \
                StateVector<Prec, Space>(Kokkos::subview(states._raw, i, Kokkos::ALL));            \
            _gate_list[indices[i]]->update_quantum_state(state_vector);                            \
            states.set_state_vector_at(i, state_vector);                                           \
        }                                                                                          \
    }
DEFINE_PROBABILISTIC_GATE_UPDATE(ExecutionSpace::Host)
DEFINE_PROBABILISTIC_GATE_UPDATE(ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_PROBABILISTIC_GATE_UPDATE(ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_PROBABILISTIC_GATE_UPDATE
template class ProbabilisticGateImpl<Prec>;

template <Precision Prec>
std::shared_ptr<const ProbabilisticGateImpl<Prec>>
GetGateFromJson<ProbabilisticGateImpl<Prec>>::get(const Json& j) {
    return std::make_shared<const ProbabilisticGateImpl<Prec>>(
        j.at("distribution").get<std::vector<double>>(),
        j.at("gate_list").get<std::vector<Gate<Prec>>>());
}
template class GetGateFromJson<ProbabilisticGateImpl<Prec>>;
}  // namespace scaluq::internal
