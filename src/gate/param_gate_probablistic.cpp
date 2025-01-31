#include <scaluq/gate/param_gate_probablistic.hpp>

#include "../util/template.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
template <Precision Prec>
ParamProbablisticGateImpl<Prec>::ParamProbablisticGateImpl(
    const std::vector<double>& distribution,
    const std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>>& gate_list)
    : ParamGateBase<Prec>(0, 0), _distribution(distribution), _gate_list(gate_list) {
=======
FLOAT_AND_SPACE(Fp, Sp)
ParamProbablisticGateImpl<Fp, Sp>::ParamProbablisticGateImpl(
    const std::vector<Fp>& distribution,
    const std::vector<std::variant<Gate<Fp, Sp>, ParamGate<Fp, Sp>>>& gate_list)
    : ParamGateBase<Fp, Sp>(0, 0), _distribution(distribution), _gate_list(gate_list) {
>>>>>>> set-space
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
<<<<<<< HEAD
template <Precision Prec>
std::shared_ptr<const ParamGateBase<Prec>> ParamProbablisticGateImpl<Prec>::get_inverse() const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::shared_ptr<const ParamGateBase<Fp, Sp>> ParamProbablisticGateImpl<Fp, Sp>::get_inverse()
    const {
>>>>>>> set-space
    std::vector<EitherGate> inv_gate_list;
    inv_gate_list.reserve(_gate_list.size());
    std::ranges::transform(
        _gate_list, std::back_inserter(inv_gate_list), [](const EitherGate& gate) {
            return std::visit([](const auto& g) { return EitherGate{g->get_inverse()}; }, gate);
        });
    return std::make_shared<const ParamProbablisticGateImpl>(_distribution, inv_gate_list);
}
<<<<<<< HEAD
template <Precision Prec>
void ParamProbablisticGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector,
                                                           double param) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ParamProbablisticGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector,
                                                             Fp param) const {
>>>>>>> set-space
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
<<<<<<< HEAD
template <Precision Prec>
void ParamProbablisticGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states,
                                                           std::vector<double> params) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ParamProbablisticGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                             std::vector<Fp> params) const {
>>>>>>> set-space
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
<<<<<<< HEAD
        auto state_vector = StateVector<Prec>(Kokkos::subview(states._raw, i, Kokkos::ALL));
=======
        auto state_vector = StateVector<Fp, Sp>(Kokkos::subview(states._raw, i, Kokkos::ALL));
>>>>>>> set-space
        if (gate.index() == 0) {
            std::get<0>(gate)->update_quantum_state(state_vector);
        } else {
            std::get<1>(gate)->update_quantum_state(state_vector, params[i]);
        }
    }
}
<<<<<<< HEAD
template <Precision Prec>
std::string ParamProbablisticGateImpl<Prec>::to_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamProbablisticGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
>>>>>>> set-space
    std::ostringstream ss;
    const auto dist = distribution();
    ss << indent << "Gate Type: Probablistic\n";
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
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(ParamProbablisticGateImpl)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(ParamProbablisticGateImpl)
>>>>>>> set-space
}  // namespace scaluq::internal
