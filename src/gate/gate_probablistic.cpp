#include <scaluq/gate/gate_probablistic.hpp>

#include "../util/template.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
template <Precision Prec>
ProbablisticGateImpl<Prec>::ProbablisticGateImpl(const std::vector<double>& distribution,
                                                 const std::vector<Gate<Prec>>& gate_list)
    : GateBase<Prec>(0, 0), _distribution(distribution), _gate_list(gate_list) {
=======
FLOAT_AND_SPACE(Fp, Sp)
ProbablisticGateImpl<Fp, Sp>::ProbablisticGateImpl(const std::vector<Fp>& distribution,
                                                   const std::vector<Gate<Fp, Sp>>& gate_list)
    : GateBase<Fp, Sp>(0, 0), _distribution(distribution), _gate_list(gate_list) {
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
std::shared_ptr<const GateBase<Prec>> ProbablisticGateImpl<Prec>::get_inverse() const {
    std::vector<Gate<Prec>> inv_gate_list;
    inv_gate_list.reserve(_gate_list.size());
    std::ranges::transform(_gate_list,
                           std::back_inserter(inv_gate_list),
                           [](const Gate<Prec>& gate) { return gate->get_inverse(); });
    return std::make_shared<const ProbablisticGateImpl>(_distribution, inv_gate_list);
}
template <Precision Prec>
void ProbablisticGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::shared_ptr<const GateBase<Fp, Sp>> ProbablisticGateImpl<Fp, Sp>::get_inverse() const {
    std::vector<Gate<Fp, Sp>> inv_gate_list;
    inv_gate_list.reserve(_gate_list.size());
    std::ranges::transform(_gate_list,
                           std::back_inserter(inv_gate_list),
                           [](const Gate<Fp, Sp>& gate) { return gate->get_inverse(); });
    return std::make_shared<const ProbablisticGateImpl<Fp, Sp>>(_distribution, inv_gate_list);
}
FLOAT_AND_SPACE(Fp, Sp)
void ProbablisticGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
>>>>>>> set-space
    Random random;
    double r = random.uniform();
    std::uint64_t i = std::distance(_cumulative_distribution.begin(),
                                    std::ranges::upper_bound(_cumulative_distribution, r)) -
                      1;
    if (i >= _gate_list.size()) i = _gate_list.size() - 1;
    _gate_list[i]->update_quantum_state(state_vector);
}
<<<<<<< HEAD
template <Precision Prec>
void ProbablisticGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ProbablisticGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
>>>>>>> set-space
    std::vector<std::uint64_t> indices(states.batch_size());
    std::vector<double> r(states.batch_size());

    Random random;
    for (std::size_t i = 0; i < states.batch_size(); ++i) {
        r[i] = random.uniform();
        indices[i] = std::distance(_cumulative_distribution.begin(),
                                   std::ranges::upper_bound(_cumulative_distribution, r[i])) -
                     1;
        if (indices[i] >= _gate_list.size()) indices[i] = _gate_list.size() - 1;
<<<<<<< HEAD
        auto state_vector = StateVector<Prec>(Kokkos::subview(states._raw, i, Kokkos::ALL));
=======
        auto state_vector = StateVector<Fp, Sp>(Kokkos::subview(states._raw, i, Kokkos::ALL));
>>>>>>> set-space
        _gate_list[indices[i]]->update_quantum_state(state_vector);
        Kokkos::parallel_for(
            "update_states", states.dim(), KOKKOS_CLASS_LAMBDA(const int j) {
                states._raw(i, j) = state_vector._raw(j);
            });
    }
}
<<<<<<< HEAD
template <Precision Prec>
std::string ProbablisticGateImpl<Prec>::to_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string ProbablisticGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
>>>>>>> set-space
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
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(ProbablisticGateImpl)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(ProbablisticGateImpl)
>>>>>>> set-space
}  // namespace scaluq::internal
