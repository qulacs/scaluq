#include <scaluq/gate/gate_probabilistic.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
namespace {
template <Precision Prec, ExecutionSpace Space>
std::uint64_t select_probabilistic_gate_index(const std::vector<double>& cumulative_distribution,
                                              ExecutionContext<Prec, Space> context) {
    std::uniform_real_distribution<double> dist(0., 1.);
    std::uint64_t i = std::distance(cumulative_distribution.begin(),
                                    std::ranges::upper_bound(cumulative_distribution,
                                                             dist(context.random_engine))) -
                      1;
    return std::min<std::uint64_t>(i, cumulative_distribution.size() - 2);
}

template <Precision Prec>
void flatten_probabilistic_gate(double prob_prefix,
                                const Gate<Prec>& gate,
                                std::vector<double>& accumulated_distribution,
                                std::vector<Gate<Prec>>& accumulated_gate_list) {
    if (gate.gate_type() != GateType::Probabilistic) {
        accumulated_distribution.push_back(prob_prefix);
        accumulated_gate_list.push_back(gate);
        return;
    }
    auto probabilistic_gate = ProbabilisticGate<Prec>(gate);
    const auto& distribution = probabilistic_gate->distribution();
    const auto& gate_list = probabilistic_gate->gate_list();
    for (std::size_t i = 0; i < distribution.size(); ++i) {
        flatten_probabilistic_gate(prob_prefix * distribution[i],
                                   gate_list[i],
                                   accumulated_distribution,
                                   accumulated_gate_list);
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
#define DEFINE_PROBABILISTIC_GATE_CONTEXT_UPDATE(Space)                                           \
    template <Precision Prec>                                                                     \
    void ProbabilisticGateImpl<Prec>::update_quantum_state(ExecutionContext<Prec, Space> context) \
        const {                                                                                   \
        const std::uint64_t i =                                                                   \
            select_probabilistic_gate_index(_cumulative_distribution, context);                   \
        _gate_list[i]->update_quantum_state(context);                                             \
    }                                                                                             \
    template <Precision Prec>                                                                     \
    void ProbabilisticGateImpl<Prec>::update_quantum_state(                                       \
        ExecutionContextBatched<Prec, Space> context) const {                                     \
        for (std::size_t i = 0; i < context.states.batch_size(); ++i) {                           \
            auto state_vector = context.states.view_state_vector_at(i);                           \
            this->update_quantum_state(                                                           \
                ExecutionContext<Prec, Space>{state_vector,                                       \
                                              context.classical_register[i],                      \
                                              context.random_engine});                            \
        }                                                                                         \
    }
DEFINE_PROBABILISTIC_GATE_CONTEXT_UPDATE(ExecutionSpace::Host)
DEFINE_PROBABILISTIC_GATE_CONTEXT_UPDATE(ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_PROBABILISTIC_GATE_CONTEXT_UPDATE(ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_PROBABILISTIC_GATE_CONTEXT_UPDATE

template class ProbabilisticGateImpl<Prec>;

template <Precision Prec>
std::shared_ptr<const ProbabilisticGateImpl<Prec>>
GetGateFromJson<ProbabilisticGateImpl<Prec>>::get(const Json& j) {
    return std::make_shared<const ProbabilisticGateImpl<Prec>>(
        j.at("distribution").get<std::vector<double>>(),
        j.at("gate_list").get<std::vector<Gate<Prec>>>());
}
template struct GetGateFromJson<ProbabilisticGateImpl<Prec>>;
}  // namespace scaluq::internal
