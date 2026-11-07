#pragma once

#include <limits>
#include <map>

#include "../util/utility.hpp"
#include "pauli_operator.hpp"

namespace scaluq {

// Immutable snapshot of a Pauli sum grouped by its bit-flip mask. cache_budget is
// measured in bytes; if the complete diagonal cache does not fit, weights are
// evaluated from the terms while applying the operator. Cached construction costs
// O(L * 2^n), and cached applications cost O(G * 2^n). Streaming costs
// O(L * 2^n). The cache budget covers the diagonal arrays, not term metadata.
template <Precision Prec, ExecutionSpace Space>
class GroupedPauliOperator {
    using Complex = internal::Complex<Prec>;
    using Exec = internal::SpaceType<Space>;
    struct Group {
        std::uint64_t x, begin, end;
    };
    struct Term {
        std::uint64_t z;
        Complex factor;
    };
    Kokkos::View<Group*, Exec> groups_;
    Kokkos::View<Term*, Exec> terms_;
    Kokkos::View<Complex**, Exec> cache_;
    std::uint64_t n_qubits_, dim_;

    struct CacheWeights {
        decltype(groups_) groups;
        decltype(terms_) terms;
        decltype(cache_) cache;

        KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t g, std::uint64_t b) const {
            Complex sum = 0.;
            for (auto t = groups(g).begin; t < groups(g).end; ++t) {
                auto term = terms(t);
                sum += (Kokkos::popcount(b & term.z) & 1) ? -term.factor : term.factor;
            }
            cache(g, b) = sum;
        }
    };

    KOKKOS_INLINE_FUNCTION static Complex weight(std::uint64_t g,
                                                 std::uint64_t b,
                                                 decltype(groups_) groups,
                                                 decltype(terms_) terms,
                                                 decltype(cache_) cache) {
        if (cache.extent(0)) return cache(g, b);
        Complex sum = 0.;
        for (auto t = groups(g).begin; t < groups(g).end; ++t) {
            auto term = terms(t);
            sum += (Kokkos::popcount(b & term.z) & 1) ? -term.factor : term.factor;
        }
        return sum;
    }
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
        groups_ = internal::convert_vector_to_view<Group, Space>(groups);
        terms_ = internal::convert_vector_to_view<Term, Space>(terms);
        if (!groups.empty() && dim_ <= cache_budget / sizeof(Complex) / groups.size()) {
            cache_ = decltype(cache_)("grouped_pauli_cache", groups.size(), dim_);
            Kokkos::parallel_for(
                "GroupedPauliOperator::cache",
                Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>(
                    {0, 0},
                    {static_cast<std::int64_t>(groups.size()), static_cast<std::int64_t>(dim_)}),
                CacheWeights{groups_, terms_, cache_});
        }
    }
    [[nodiscard]] std::uint64_t n_groups() const { return groups_.extent(0); }
    [[nodiscard]] bool is_cached() const { return cache_.extent(0) != 0; }
    // Add coef * H * input to output. Aliased inputs are copied once.
    void apply_to_state(const StateVector<Prec, Space>& input,
                        StateVector<Prec, Space>& output,
                        StdComplex coef = 1.) const {
        check(input);
        check(output);
        auto source = input._raw.data() == output._raw.data() ? input.copy() : input;
        auto gs = groups_;
        auto ts = terms_;
        auto cache = cache_;
        auto ng = n_groups();
        Complex factor(coef);
        Kokkos::parallel_for(
            "GroupedPauliOperator::apply",
            Kokkos::RangePolicy<Exec>(0, dim_),
            KOKKOS_LAMBDA(std::uint64_t i) {
                Complex sum = 0.;
                for (std::uint64_t g = 0; g < ng; ++g) {
                    auto b = i ^ gs(g).x;
                    sum += weight(g, b, gs, ts, cache) * source.at_unsafe(b);
                }
                output.at_unsafe(i) += factor * sum;
            });
    }
    [[nodiscard]] StdComplex get_transition_amplitude(const StateVector<Prec, Space>& bra,
                                                      const StateVector<Prec, Space>& ket) const {
        check(bra);
        check(ket);
        auto gs = groups_;
        auto ts = terms_;
        auto cache = cache_;
        auto ng = n_groups();
        Complex result;
        Kokkos::parallel_reduce(
            "GroupedPauliOperator::transition",
            Kokkos::RangePolicy<Exec>(0, dim_),
            KOKKOS_LAMBDA(std::uint64_t b, Complex& sum) {
                for (std::uint64_t g = 0; g < ng; ++g)
                    sum += internal::conj(bra.at_unsafe(b ^ gs(g).x)) *
                           weight(g, b, gs, ts, cache) * ket.at_unsafe(b);
            },
            internal::Sum<Complex, Space>(result));
        return static_cast<StdComplex>(result);
    }
    [[nodiscard]] StdComplex get_expectation_value(const StateVector<Prec, Space>& state) const {
        return get_transition_amplitude(state, state);
    }
};
}  // namespace scaluq
