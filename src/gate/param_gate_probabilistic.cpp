#include <scaluq/gate/param_gate_probabilistic.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
namespace {
template <Precision Prec>
using EitherGate = std::variant<Gate<Prec>, ParamGate<Prec>>;

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
template <Precision Prec>
void ParamProbabilisticGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Host>& state_vector, double param) const {
    Random random;
    double r = random.uniform();
    std::uint64_t i = std::distance(_cumulative_distribution.begin(),
                                    std::ranges::upper_bound(_cumulative_distribution, r)) -
                      1;
    if (i >= _gate_list.size()) i = _gate_list.size() - 1;
    const auto& gate = _gate_list[i];
    if (gate.index() == 0) {
        std::get<0>(gate)->update_quantum_state(state_vector);
    } else {
        std::get<1>(gate)->update_quantum_state(state_vector, param);
    }
}
template <Precision Prec>
void ParamProbabilisticGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Host>& states, std::vector<double> params) const {
    Random random;
    std::vector<double> r(states.batch_size());
    std::ranges::generate(r, [&random]() { return random.uniform(); });
    std::vector<std::uint64_t> indicies(states.batch_size());
    std::ranges::transform(r, indicies.begin(), [this](double r) {
        return std::distance(_cumulative_distribution.begin(),
                             std::ranges::upper_bound(_cumulative_distribution, r)) -
               1;
    });
    std::ranges::transform(indicies, indicies.begin(), [this](std::uint64_t i) {
        if (i >= _gate_list.size()) i = _gate_list.size() - 1;
        return i;
    });
    for (std::size_t i = 0; i < states.batch_size(); ++i) {
        const auto& gate = _gate_list[indicies[i]];
        auto state_vector = states.view_state_vector_at(i);
        if (gate.index() == 0) {
            std::get<0>(gate)->update_quantum_state(state_vector);
        } else {
            std::get<1>(gate)->update_quantum_state(state_vector, params[i]);
        }
    }
}
template <Precision Prec>
void ParamProbabilisticGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::HostSerial>& state_vector, double param) const {
    Random random;
    double r = random.uniform();
    std::uint64_t i = std::distance(_cumulative_distribution.begin(),
                                    std::ranges::upper_bound(_cumulative_distribution, r)) -
                      1;
    if (i >= _gate_list.size()) i = _gate_list.size() - 1;
    const auto& gate = _gate_list[i];
    if (gate.index() == 0) {
        std::get<0>(gate)->update_quantum_state(state_vector);
    } else {
        std::get<1>(gate)->update_quantum_state(state_vector, param);
    }
}
template <Precision Prec>
void ParamProbabilisticGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
    std::vector<double> params) const {
    Random random;
    std::vector<double> r(states.batch_size());
    std::ranges::generate(r, [&random]() { return random.uniform(); });
    std::vector<std::uint64_t> indicies(states.batch_size());
    std::ranges::transform(r, indicies.begin(), [this](double r) {
        return std::distance(_cumulative_distribution.begin(),
                             std::ranges::upper_bound(_cumulative_distribution, r)) -
               1;
    });
    std::ranges::transform(indicies, indicies.begin(), [this](std::uint64_t i) {
        if (i >= _gate_list.size()) i = _gate_list.size() - 1;
        return i;
    });
    for (std::size_t i = 0; i < states.batch_size(); ++i) {
        const auto& gate = _gate_list[indicies[i]];
        auto state_vector = states.view_state_vector_at(i);
        if (gate.index() == 0) {
            std::get<0>(gate)->update_quantum_state(state_vector);
        } else {
            std::get<1>(gate)->update_quantum_state(state_vector, params[i]);
        }
    }
}
#ifdef SCALUQ_USE_CUDA
template <Precision Prec>
void ParamProbabilisticGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Default>& state_vector, double param) const {
    Random random;
    double r = random.uniform();
    std::uint64_t i = std::distance(_cumulative_distribution.begin(),
                                    std::ranges::upper_bound(_cumulative_distribution, r)) -
                      1;
    if (i >= _gate_list.size()) i = _gate_list.size() - 1;
    const auto& gate = _gate_list[i];
    if (gate.index() == 0) {
        std::get<0>(gate)->update_quantum_state(state_vector);
    } else {
        std::get<1>(gate)->update_quantum_state(state_vector, param);
    }
}
template <Precision Prec>
void ParamProbabilisticGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Default>& states, std::vector<double> params) const {
    Random random;
    std::vector<double> r(states.batch_size());
    std::ranges::generate(r, [&random]() { return random.uniform(); });
    std::vector<std::uint64_t> indicies(states.batch_size());
    std::ranges::transform(r, indicies.begin(), [this](double r) {
        return std::distance(_cumulative_distribution.begin(),
                             std::ranges::upper_bound(_cumulative_distribution, r)) -
               1;
    });
    std::ranges::transform(indicies, indicies.begin(), [this](std::uint64_t i) {
        if (i >= _gate_list.size()) i = _gate_list.size() - 1;
        return i;
    });
    for (std::size_t i = 0; i < states.batch_size(); ++i) {
        const auto& gate = _gate_list[indicies[i]];
        auto state_vector = states.view_state_vector_at(i);
        if (gate.index() == 0) {
            std::get<0>(gate)->update_quantum_state(state_vector);
        } else {
            std::get<1>(gate)->update_quantum_state(state_vector, params[i]);
        }
    }
}
#endif
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
template class GetParamGateFromJson<ParamProbabilisticGateImpl<Prec>>;
}  // namespace scaluq::internal
