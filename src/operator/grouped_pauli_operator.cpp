#include <bit>
#include <map>
#include <scaluq/operator/grouped_pauli_operator.hpp>
#include <scaluq/prec_space.hpp>
#include <stdexcept>

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
void GroupedPauliOperator<Prec, Space>::check(const StateVector<Prec, Space>& state) const {
    if (state.n_qubits() != n_qubits_) {
        throw std::invalid_argument("GroupedPauliOperator: incompatible qubit count");
    }
}

template <Precision Prec, ExecutionSpace Space>
GroupedPauliOperator<Prec, Space>::GroupedPauliOperator(
    std::uint64_t n_qubits,
    const std::vector<PauliOperator<Prec>>& input,
    std::uint64_t cache_budget)
    : n_qubits_(n_qubits) {
    if (n_qubits >= 63) {
        throw std::invalid_argument("GroupedPauliOperator: too many qubits");
    }
    dim_ = 1ULL << n_qubits;
    using Group = internal::GroupedPauliGroup;
    using Term = internal::GroupedPauliTerm<Prec>;
    std::map<std::uint64_t, std::vector<Term>> grouped;
    for (const auto& p : input) {
        if (static_cast<std::uint64_t>(std::bit_width(p._bit_flip_mask | p._phase_flip_mask)) >
            n_qubits) {
            throw std::invalid_argument("GroupedPauliOperator: insufficient qubits");
        }
        auto [x, z] = p.get_XZ_mask_representation();
        grouped[x].push_back(
            {z, p._coef * internal::PHASE_90ROT<Prec>()[std::popcount(x & z) % 4]});
    }
    std::vector<Group> groups;
    std::vector<Term> terms;
    for (const auto& [x, values] : grouped) {
        auto begin = terms.size();
        terms.insert(terms.end(), values.begin(), values.end());
        groups.push_back({x, begin, terms.size()});
    }
    weights_.groups = internal::convert_vector_to_view<Group, Space>(groups);
    weights_.terms = internal::convert_vector_to_view<Term, Space>(terms);
    if (!groups.empty() && dim_ <= cache_budget / sizeof(Complex) / groups.size()) {
        weights_.cache = decltype(weights_.cache)(
            Kokkos::ViewAllocateWithoutInitializing("grouped_pauli_cache"), groups.size(), dim_);
        Kokkos::parallel_for(
            "GroupedPauliOperator::cache",
            Kokkos::MDRangePolicy<Exec,
                                  Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
                {0, 0},
                {static_cast<std::int64_t>(groups.size()), static_cast<std::int64_t>(dim_)}),
            weights_);
    }
}

template <Precision Prec, ExecutionSpace Space>
void GroupedPauliOperator<Prec, Space>::apply_add(const StateVector<Prec, Space>& input,
                                                  StateVector<Prec, Space>& output,
                                                  StdComplex coef) const {
    check(input);
    check(output);
    auto ng = n_groups();
    if (ng == 0) return;
    auto weights = weights_;
    internal::Complex<Prec> factor(coef);
    Kokkos::parallel_for(
        "GroupedPauliOperator::apply_add",
        Kokkos::RangePolicy<Exec>(0, dim_),
        KOKKOS_LAMBDA(std::uint64_t i) {
            Complex sum = 0.;
            for (std::uint64_t g = 0; g < ng; ++g) {
                auto b = i ^ weights.groups(g).x;
                sum += weights.get(g, b) * input.at_unsafe(b);
            }
            output.at_unsafe(i) += factor * sum;
        });
}

template <Precision Prec, ExecutionSpace Space>
StdComplex GroupedPauliOperator<Prec, Space>::get_transition_amplitude(
    const StateVector<Prec, Space>& bra, const StateVector<Prec, Space>& ket) const {
    check(bra);
    check(ket);
    auto ng = n_groups();
    if (ng == 0) return StdComplex(0.);
    auto weights = weights_;
    Kokkos::complex<double> result{};
    Kokkos::parallel_reduce(
        "GroupedPauliOperator::transition",
        Kokkos::RangePolicy<Exec>(0, dim_),
        KOKKOS_LAMBDA(std::uint64_t b, Kokkos::complex<double>& sum) {
            for (std::uint64_t g = 0; g < ng; ++g) {
                const Complex term = internal::conj(bra.at_unsafe(b ^ weights.groups(g).x)) *
                                     weights.get(g, b) * ket.at_unsafe(b);
                sum += internal::to_complex_double(term);
            }
        },
        internal::Sum<Kokkos::complex<double>, Space>(result));
    return static_cast<StdComplex>(result);
}

template <Precision Prec, ExecutionSpace Space>
StdComplex GroupedPauliOperator<Prec, Space>::get_expectation_value(
    const StateVector<Prec, Space>& state) const {
    return get_transition_amplitude(state, state);
}

template class GroupedPauliOperator<internal::Prec, internal::Space>;

}  // namespace scaluq
