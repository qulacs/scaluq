#pragma once

#include <limits>
#include <map>

#include "../util/utility.hpp"
#include "pauli_operator.hpp"

namespace scaluq {

namespace internal {

// CUDA kernels need accessible types for their captured weights and views.
struct GroupedPauliGroup {
    std::uint64_t x, begin, end;
};

template <Precision Prec>
struct GroupedPauliTerm {
    std::uint64_t z;
    Complex<Prec> factor;
};

template <Precision Prec, ExecutionSpace Space>
struct GroupedPauliWeights {
    using Exec = SpaceType<Space>;
    Kokkos::View<GroupedPauliGroup*, Exec> groups;
    Kokkos::View<GroupedPauliTerm<Prec>*, Exec> terms;
    Kokkos::View<Complex<Prec>**, Exec> cache;

    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t g, std::uint64_t b) const {
        cache(g, b) = evaluate(g, b);
    }

    KOKKOS_INLINE_FUNCTION Complex<Prec> get(std::uint64_t g, std::uint64_t b) const {
        if (cache.extent(0)) return cache(g, b);
        return evaluate(g, b);
    }

private:
    KOKKOS_INLINE_FUNCTION Complex<Prec> evaluate(std::uint64_t g, std::uint64_t b) const {
        Complex<Prec> sum = 0.;
        for (auto t = groups(g).begin; t < groups(g).end; ++t) {
            auto term = terms(t);
            sum += (Kokkos::popcount(b & term.z) & 1) ? -term.factor : term.factor;
        }
        return sum;
    }
};

}  // namespace internal

// Immutable snapshot of a Pauli sum grouped by its bit-flip mask. cache_budget is
// measured in bytes; if the complete diagonal cache does not fit, weights are
// evaluated from the terms while applying the operator. Cached construction costs
// O(L * 2^n), and cached applications cost O(G * 2^n). Streaming costs
// O(L * 2^n). The cache budget covers the diagonal arrays, not term metadata.
template <Precision Prec, ExecutionSpace Space>
class GroupedPauliOperator {
    using Complex = internal::Complex<Prec>;
    using Exec = internal::SpaceType<Space>;
    internal::GroupedPauliWeights<Prec, Space> weights_;
    std::uint64_t n_qubits_, dim_;

    void check(const StateVector<Prec, Space>& state) const {
        if (state.n_qubits() != n_qubits_)
            throw std::runtime_error("GroupedPauliOperator: incompatible qubit count");
    }

public:
    GroupedPauliOperator(std::uint64_t n_qubits,
                         const std::vector<PauliOperator<Prec>>& input,
                         std::uint64_t cache_budget = 0)
        : n_qubits_(n_qubits) {
        if (n_qubits >= 63) throw std::runtime_error("GroupedPauliOperator: too many qubits");
        dim_ = 1ULL << n_qubits;
        using Group = internal::GroupedPauliGroup;
        using Term = internal::GroupedPauliTerm<Prec>;
        std::map<std::uint64_t, std::vector<Term>> grouped;
        for (const auto& p : input) {
            if (static_cast<std::uint64_t>(std::bit_width(p._bit_flip_mask | p._phase_flip_mask)) >
                n_qubits)
                throw std::runtime_error("GroupedPauliOperator: insufficient qubits");
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
            weights_.cache = decltype(weights_.cache)("grouped_pauli_cache", groups.size(), dim_);
            Kokkos::parallel_for(
                "GroupedPauliOperator::cache",
                Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>(
                    {0, 0},
                    {static_cast<std::int64_t>(groups.size()), static_cast<std::int64_t>(dim_)}),
                weights_);
        }
    }
    [[nodiscard]] std::uint64_t n_groups() const { return weights_.groups.extent(0); }
    [[nodiscard]] bool is_cached() const { return weights_.cache.extent(0) != 0; }
    // Add coef * H * input to output. input and output must not overlap; any overlap,
    // including passing the same state vector twice, is undefined behavior.
    void apply_to_state(const StateVector<Prec, Space>& input,
                        StateVector<Prec, Space>& output,
                        StdComplex coef = 1.) const {
        check(input);
        check(output);
        auto weights = weights_;
        auto ng = n_groups();
        internal::Complex<Prec> factor(coef);
        Kokkos::parallel_for(
            "GroupedPauliOperator::apply",
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
    [[nodiscard]] StdComplex get_transition_amplitude(const StateVector<Prec, Space>& bra,
                                                      const StateVector<Prec, Space>& ket) const {
        check(bra);
        check(ket);
        auto weights = weights_;
        auto ng = n_groups();
        Complex result;
        Kokkos::parallel_reduce(
            "GroupedPauliOperator::transition",
            Kokkos::RangePolicy<Exec>(0, dim_),
            KOKKOS_LAMBDA(std::uint64_t b, internal::Complex<Prec>& sum) {
                for (std::uint64_t g = 0; g < ng; ++g)
                    sum += internal::conj(bra.at_unsafe(b ^ weights.groups(g).x)) *
                           weights.get(g, b) * ket.at_unsafe(b);
            },
            internal::Sum<Complex, Space>(result));
        return static_cast<StdComplex>(result);
    }
    [[nodiscard]] StdComplex get_expectation_value(const StateVector<Prec, Space>& state) const {
        return get_transition_amplitude(state, state);
    }
};
}  // namespace scaluq
