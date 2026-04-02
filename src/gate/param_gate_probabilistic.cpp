#include <scaluq/gate/param_gate_probabilistic.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
namespace {
template <Precision Prec>
using EitherGate = std::variant<Gate<Prec>, ParamGate<Prec>>;

template <Precision Prec, ExecutionSpace Space>
std::uint64_t select_param_probabilistic_gate_index(
    const std::vector<double>& cumulative_distribution,
    ExecutionContext<Prec, Space> context) {
    if (context.rng == nullptr) {
        throw std::runtime_error(
            "ParamProbabilisticGateImpl::update_quantum_state(): random engine is required.");
    }

    std::uniform_real_distribution<double> dist(0., 1.);
    const std::uint64_t i =
        std::distance(cumulative_distribution.begin(),
                      std::ranges::upper_bound(cumulative_distribution, dist(*context.rng))) -
        1;
    return std::min<std::uint64_t>(i, cumulative_distribution.size() - 2);
}

template <Precision Prec, ExecutionSpace Space>
void apply_param_probabilistic_gate_once(const std::vector<double>& cumulative_distribution,
                                         const std::vector<EitherGate<Prec>>& gate_list,
                                         ExecutionContext<Prec, Space> context,
                                         double param) {
    const auto& gate =
        gate_list[select_param_probabilistic_gate_index(cumulative_distribution, context)];
    if (gate.index() == 0) {
        std::get<0>(gate)->update_quantum_state(context);
    } else {
        std::get<1>(gate)->update_quantum_state(context, param);
    }
}

template <Precision Prec>
void flatten_param_probabilistic_gate(double prob_prefix,
                                      const EitherGate<Prec>& gate,
                                      std::vector<double>& accumulated_distribution,
                                      std::vector<EitherGate<Prec>>& accumulated_gate_list) {
    if (gate.index() == 0) {
        const auto& gate_non_param = std::get<0>(gate);
        if (gate_non_param.gate_type() == GateType::Probabilistic) {
            auto probabilistic_gate = ProbabilisticGate<Prec>(gate_non_param);
            const auto& distribution = probabilistic_gate->distribution();
            const auto& gate_list = probabilistic_gate->gate_list();
            for (std::size_t i = 0; i < distribution.size(); ++i) {
                flatten_param_probabilistic_gate(prob_prefix * distribution[i],
                                                 EitherGate<Prec>{gate_list[i]},
                                                 accumulated_distribution,
                                                 accumulated_gate_list);
            }
            return;
        }
    } else {
        const auto& gate_param = std::get<1>(gate);
        if (gate_param.param_gate_type() == ParamGateType::ParamProbabilistic) {
            auto probabilistic_gate = ParamProbabilisticGate<Prec>(gate_param);
            const auto& distribution = probabilistic_gate->distribution();
            const auto& gate_list = probabilistic_gate->gate_list();
            for (std::size_t i = 0; i < distribution.size(); ++i) {
                flatten_param_probabilistic_gate(prob_prefix * distribution[i],
                                                 gate_list[i],
                                                 accumulated_distribution,
                                                 accumulated_gate_list);
            }
            return;
        }
    }
    accumulated_distribution.push_back(prob_prefix);
    accumulated_gate_list.push_back(gate);
}
}  // namespace

template <Precision Prec>
ParamProbabilisticGateImpl<Prec>::ParamProbabilisticGateImpl(
    const std::vector<double>& distribution,
    const std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>>& gate_list)
    : ParamGateBase<Prec>(0, 0, 0) {
    std::uint64_t n = distribution.size();
    if (n == 0) {
        throw std::runtime_error("At least one gate is required.");
    }
    if (n != gate_list.size()) {
        throw std::runtime_error("distribution and gate_list have different size.");
    }

    for (std::size_t i = 0; i < n; ++i) {
        flatten_param_probabilistic_gate(distribution[i], gate_list[i], _distribution, _gate_list);
    }

    _cumulative_distribution.resize(_distribution.size() + 1);
    std::partial_sum(
        _distribution.begin(), _distribution.end(), _cumulative_distribution.begin() + 1);
    if (std::abs(_cumulative_distribution.back() - 1.) > 1e-6) {
        throw std::runtime_error("Sum of distribution must be equal to 1.");
    }
}
template <Precision Prec>
std::shared_ptr<const ParamGateBase<Prec>> ParamProbabilisticGateImpl<Prec>::get_inverse() const {
    std::vector<EitherGate> inv_gate_list;
    inv_gate_list.reserve(_gate_list.size());
    std::ranges::transform(
        _gate_list, std::back_inserter(inv_gate_list), [](const EitherGate& gate) {
            return std::visit([](const auto& g) { return EitherGate{g->get_inverse()}; }, gate);
        });
    return std::make_shared<const ParamProbabilisticGateImpl>(_distribution, inv_gate_list);
}
template <Precision Prec>
std::string ParamProbabilisticGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    const auto dist = distribution();
    ss << indent << "Gate Type: ParamProbabilistic\n";
    for (std::size_t i = 0; i < dist.size(); ++i) {
        ss << indent << "  --------------------\n";
        ss << indent << "  Probability: " << dist[i] << "\n";
        std::visit(
            [&](auto&& arg) {
                ss << arg->to_string(indent + "  ") << (i == dist.size() - 1 ? "" : "\n");
            },
            gate_list()[i]);
    }
    return ss.str();
}
#define DEFINE_PARAM_PROBABILISTIC_GATE_UPDATE(Space)                                      \
    template <Precision Prec>                                                               \
    void ParamProbabilisticGateImpl<Prec>::update_quantum_state(                            \
        ExecutionContext<Prec, Space> context, double param) const {                        \
        apply_param_probabilistic_gate_once(_cumulative_distribution, _gate_list, context,  \
                                            param);                                          \
    }                                                                                       \
    template <Precision Prec>                                                               \
    void ParamProbabilisticGateImpl<Prec>::update_quantum_state(                            \
        BatchedExecutionContext<Prec, Space> context, std::vector<double> params) const {   \
        if (params.size() != context.states.batch_size()) {                                 \
            throw std::runtime_error(                                                       \
                "ParamProbabilisticGateImpl::update_quantum_state(): params size must "     \
                "match batch size.");                                                       \
        }                                                                                   \
        if (context.reg != nullptr &&                                                      \
            context.reg->size() != context.n_classical_bits * context.states.batch_size()) { \
            throw std::runtime_error(                                                       \
                "ParamProbabilisticGateImpl::update_quantum_state(): classical register "    \
                "size must match n_classical_bits * batch size.");                          \
        }                                                                                   \
        for (std::size_t i = 0; i < context.states.batch_size(); ++i) {                     \
            auto state_vector = context.states.view_state_vector_at(i);                      \
            apply_param_probabilistic_gate_once(                                            \
                _cumulative_distribution,                                                   \
                _gate_list,                                                                 \
                ExecutionContext<Prec, Space>{state_vector,                                 \
                                              context.reg,                                  \
                                              context.rng,                                  \
                                              i * context.n_classical_bits},                \
                params[i]);                                                                 \
        }                                                                                   \
    }
DEFINE_PARAM_PROBABILISTIC_GATE_UPDATE(ExecutionSpace::Host)
DEFINE_PARAM_PROBABILISTIC_GATE_UPDATE(ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_PARAM_PROBABILISTIC_GATE_UPDATE(ExecutionSpace::Default)
#endif
#undef DEFINE_PARAM_PROBABILISTIC_GATE_UPDATE
template class ParamProbabilisticGateImpl<Prec>;

template <Precision Prec>
std::shared_ptr<const ParamProbabilisticGateImpl<Prec>>
GetParamGateFromJson<ParamProbabilisticGateImpl<Prec>>::get(const Json& j) {
    auto distribution = j.at("distribution").get<std::vector<double>>();
    std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>> gate_list;
    const Json& tmp_list = j.at("gate_list");
    for (const Json& tmp_j : tmp_list) {
        if (tmp_j.at("type").get<std::string>().starts_with("Param"))
            gate_list.emplace_back(tmp_j.get<ParamGate<Prec>>());
        else
            gate_list.emplace_back(tmp_j.get<Gate<Prec>>());
    }
    return std::make_shared<const ParamProbabilisticGateImpl<Prec>>(distribution, gate_list);
}
template struct GetParamGateFromJson<ParamProbabilisticGateImpl<Prec>>;
}  // namespace scaluq::internal
