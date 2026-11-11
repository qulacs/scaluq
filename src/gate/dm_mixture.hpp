#pragma once

// Deterministic application of a probabilistic mixture of operations to a density matrix:
//   ρ → Σ_k p_k E_k(ρ).
//
// Fast path: when every term is a (possibly controlled) matrix gate and all of them together
// touch at most two qubits, the whole mixture collapses into one superoperator
//   S = Σ_k p_k U_k ⊗ conj(U_k)
// on those qubits, applied in a single pass (e.g. depolarizing / bit-flip / dephasing noise).
//
// 3..6 qubits: the mixture is applied as Kraus operators √p_k U_k with scalar-identity terms
// folded into a coefficient, again in a single pass (one team per block, block in scratch).
//
// General path: each term is applied to a scratch copy of ρ and accumulated. Two scratch
// buffers are allocated regardless of the number of terms.

#include <algorithm>
#include <cmath>
#include <optional>

#include "update_ops.hpp"

namespace scaluq::internal {

struct DmMixtureTerm {
    double probability;
    std::uint64_t target_mask;
    std::uint64_t control_mask;
    std::uint64_t control_value_mask;
    ComplexMatrix matrix;  // on the target qubits only
};

// Largest operand set handled by the one-pass Kraus kernel (block of 2^6 × 2^6 in scratch).
inline constexpr int kMaxKrausMixtureQubits = 6;

// If u = λ·I, returns |λ|² (the term then contributes p|λ|²·ρ).
inline std::optional<double> identity_weight(const ComplexMatrix& u) {
    const StdComplex lambda = u(0, 0);
    const double tol = 1e-12 * std::max(1.0, std::abs(lambda));
    for (Eigen::Index i = 0; i < u.rows(); ++i)
        for (Eigen::Index j = 0; j < u.cols(); ++j)
            if (std::abs(u(i, j) - (i == j ? lambda : StdComplex(0))) > tol) return std::nullopt;
    return std::norm(lambda);
}

// 3..6 operand qubits: ρ → c·ρ + Σ_k (√p_k U_k) ρ (√p_k U_k)†, one pass over ρ.
template <Precision Prec, ExecutionSpace Space>
void apply_mixture_kraus(const std::vector<DmMixtureTerm>& terms,
                         std::uint64_t operand_mask,
                         DensityMatrix<Prec, Space>& dm) {
    std::vector<std::uint64_t> operands = mask_to_vector(operand_mask);
    const Eigen::Index n = Eigen::Index{1} << operands.size();
    double identity_coef = 0.;
    std::vector<ComplexMatrix> ops;
    for (const auto& t : terms) {
        if (t.probability == 0.) continue;
        ComplexMatrix u = get_expanded_matrix(t.matrix,
                                              mask_to_vector(t.target_mask),
                                              t.control_mask,
                                              t.control_value_mask,
                                              operands);
        if (auto w = identity_weight(u)) {
            identity_coef += t.probability * *w;
        } else {
            ops.push_back(std::sqrt(t.probability) * u);
        }
    }
    if (ops.empty()) {
        if (identity_coef != 1.) dm.multiply_coef(identity_coef);
        return;
    }
    ComplexMatrix stacked(n * static_cast<Eigen::Index>(ops.size()), n);
    for (std::size_t k = 0; k < ops.size(); ++k)
        stacked.block(static_cast<Eigen::Index>(k) * n, 0, n, n) = ops[k];
    kraus_mixture_gate(operand_mask, identity_coef, stacked, dm);
}

// Returns false (and does nothing) if no fused path applies.
template <Precision Prec, ExecutionSpace Space>
bool apply_mixture_superoperator(const std::vector<DmMixtureTerm>& terms,
                                 DensityMatrix<Prec, Space>& dm) {
    std::uint64_t operand_mask = 0;
    for (const auto& t : terms) operand_mask |= t.target_mask | t.control_mask;
    const int k = std::popcount(operand_mask);
    if (k > kMaxKrausMixtureQubits) return false;
    if (k > 2) {
        apply_mixture_kraus(terms, operand_mask, dm);
        return true;
    }

    std::vector<std::uint64_t> operands = mask_to_vector(operand_mask);
    const std::uint64_t n = 1ULL << operands.size();
    ComplexMatrix superop = ComplexMatrix::Zero(n * n, n * n);
    for (const auto& t : terms) {
        if (t.probability == 0.) continue;
        const ComplexMatrix u = get_expanded_matrix(t.matrix,
                                                    mask_to_vector(t.target_mask),
                                                    t.control_mask,
                                                    t.control_value_mask,
                                                    operands);
        // (U B U†)_{ij} = Σ_{kl} U_{ik} conj(U_{jl}) B_{kl}
        for (std::uint64_t i = 0; i < n; ++i)
            for (std::uint64_t j = 0; j < n; ++j)
                for (std::uint64_t k = 0; k < n; ++k)
                    for (std::uint64_t l = 0; l < n; ++l)
                        superop(i * n + j, k * n + l) +=
                            t.probability * u(i, k) * std::conj(u(j, l));
    }
    superoperator_gate(operand_mask, superop, dm);
    return true;
}

// apply(k, state) applies the k-th term to `state` in place.
template <Precision Prec, ExecutionSpace Space, class Apply>
void apply_mixture_general(const std::vector<double>& probabilities,
                           const Apply& apply,
                           DensityMatrix<Prec, Space>& dm) {
    std::vector<std::size_t> active;
    for (std::size_t k = 0; k < probabilities.size(); ++k)
        if (probabilities[k] != 0.) active.push_back(k);
    if (active.empty()) return;
    if (active.size() == 1) {
        apply(active[0], dm);
        if (probabilities[active[0]] != 1.) dm.multiply_coef(probabilities[active[0]]);
        return;
    }

    auto acc = DensityMatrix<Prec, Space>::uninitialized_state(dm.n_qubits());
    auto work = DensityMatrix<Prec, Space>::uninitialized_state(dm.n_qubits());
    // First term initializes the accumulator directly.
    acc.load(dm);
    apply(active[0], acc);
    acc.multiply_coef(probabilities[active[0]]);
    for (std::size_t a = 1; a + 1 < active.size(); ++a) {
        work.load(dm);
        apply(active[a], work);
        acc.add_density_matrix_with_coef(probabilities[active[a]], work);
    }
    // Last term reuses ρ itself as the scratch buffer.
    apply(active.back(), dm);
    acc.add_density_matrix_with_coef(probabilities[active.back()], dm);
    std::swap(dm._raw, acc._raw);
}

}  // namespace scaluq::internal
