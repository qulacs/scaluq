#pragma once

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
    Kokkos::View<Complex<Prec>**, Kokkos::LayoutRight, Exec> cache;

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

    void check(const StateVector<Prec, Space>& state) const;

public:
    GroupedPauliOperator(std::uint64_t n_qubits,
                         const std::vector<PauliOperator<Prec>>& input,
                         std::uint64_t cache_budget = 0);

    [[nodiscard]] std::uint64_t n_groups() const { return weights_.groups.extent(0); }
    [[nodiscard]] bool is_cached() const { return weights_.cache.extent(0) != 0; }

    // Add coef * H * input to output. input and output must not overlap; any overlap,
    // including passing the same state vector twice, is undefined behavior.
    void apply_add(const StateVector<Prec, Space>& input,
                   StateVector<Prec, Space>& output,
                   StdComplex coef = 1.) const;

    [[nodiscard]] StdComplex get_transition_amplitude(const StateVector<Prec, Space>& bra,
                                                      const StateVector<Prec, Space>& ket) const;

    [[nodiscard]] StdComplex get_expectation_value(const StateVector<Prec, Space>& state) const;
};
}  // namespace scaluq
