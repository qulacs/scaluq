#pragma once

// Building blocks for single-pass density-matrix gate updates.
//
// A gate acting on the qubits in `group_mask` (K qubits, N = 2^K) with controls (cm, cv) maps
//   ρ → U ρ U†   restricted to the subspace where the control condition holds.
// Grouping basis indices by the bits outside `group_mask`, ρ splits into N×N blocks
// B(rb, cb) = { ρ(rb ^ off[i], cb ^ off[j]) }. Each block transforms independently:
//   row & col active : B → U B U†
//   row active only  : B → U B
//   col active only  : B → B U†
//   neither          : unchanged (never visited)
// so every element is read and written at most once, in place, without temporaries.
//
// DmBlockIndexer enumerates exactly the visited blocks in a flat index space whose innermost
// dimension walks columns of a fixed row group, matching the LayoutRight storage of ρ.

#include <bit>
#include <cstdint>

#include "update_ops.hpp"

namespace scaluq::internal {

// Deposits the low bits of `value` into the set bits of `mask`, lowest first (software pdep).
// Complements insert_zero_at_mask_positions, which fills the clear bits of the mask instead.
KOKKOS_INLINE_FUNCTION std::uint64_t deposit_bits(std::uint64_t value, std::uint64_t mask) {
    std::uint64_t result = 0;
    for (std::uint64_t m = mask; m; m &= m - 1) {
        if (value & 1) result |= m & (~m + 1);
        value >>= 1;
    }
    return result;
}

// Gathers the bits of `value` at the set bits of `mask` into the low bits of the result
// (software pext); the inverse of deposit_bits.
inline std::uint64_t extract_bits(std::uint64_t value, std::uint64_t mask) {
    std::uint64_t result = 0;
    std::uint64_t out_bit = 1;
    for (std::uint64_t m = mask; m; m &= m - 1) {
        if (value & m & (~m + 1)) result |= out_bit;
        out_bit <<= 1;
    }
    return result;
}

struct DmBlockIndexer {
    std::uint64_t group_mask;
    std::uint64_t control_mask;
    std::uint64_t control_value_mask;
    std::uint64_t log_groups;      // log2(#row groups)
    std::uint64_t log_active;      // log2(#control-active row groups)
    std::uint64_t n_inactive_pat;  // 2^#controls - 1
    std::uint64_t active_pattern;  // compressed control value
    std::uint64_t region_a;        // active row groups × all column groups
    std::uint64_t total;

    DmBlockIndexer(std::uint64_t n_qubits,
                   std::uint64_t group_mask_,
                   std::uint64_t control_mask_,
                   std::uint64_t control_value_mask_)
        : group_mask(group_mask_),
          control_mask(control_mask_),
          control_value_mask(control_value_mask_ & control_mask_) {
        const std::uint64_t n_ctrl = std::popcount(control_mask);
        log_groups = n_qubits - std::popcount(group_mask);
        log_active = log_groups - n_ctrl;
        n_inactive_pat = (1ULL << n_ctrl) - 1;
        active_pattern = extract_bits(control_value_mask, control_mask);
        region_a = 1ULL << (log_active + log_groups);
        const std::uint64_t region_b = ((1ULL << log_groups) - (1ULL << log_active)) << log_active;
        total = region_a + region_b;
    }

    // Decodes flat index g into (row base, col base, row active, col active).
    KOKKOS_INLINE_FUNCTION void decode(std::uint64_t g,
                                       std::uint64_t& rb,
                                       std::uint64_t& cb,
                                       bool& row_active,
                                       bool& col_active) const {
        if (control_mask == 0) {
            rb = insert_zero_at_mask_positions(g >> log_groups, group_mask);
            cb = insert_zero_at_mask_positions(g & ((1ULL << log_groups) - 1), group_mask);
            row_active = col_active = true;
            return;
        }
        const std::uint64_t skip = group_mask | control_mask;
        if (g < region_a) {
            rb = insert_zero_at_mask_positions(g >> log_groups, skip) | control_value_mask;
            cb = insert_zero_at_mask_positions(g & ((1ULL << log_groups) - 1), group_mask);
            row_active = true;
            col_active = (cb & control_mask) == control_value_mask;
        } else {
            g -= region_a;
            const std::uint64_t r = g >> log_active;
            const std::uint64_t hi = r / n_inactive_pat;
            std::uint64_t pat = r - hi * n_inactive_pat;
            pat += (pat >= active_pattern);
            rb = insert_zero_at_mask_positions(hi, skip) | deposit_bits(pat, control_mask);
            cb = insert_zero_at_mask_positions(g & ((1ULL << log_active) - 1), skip) |
                 control_value_mask;
            row_active = false;
            col_active = true;
        }
    }
};

// Offsets (XOR) of the N members of a group whose varying bits are `group_mask`.
template <int N>
inline Kokkos::Array<std::uint64_t, N> dm_group_offsets(std::uint64_t group_mask) {
    Kokkos::Array<std::uint64_t, N> off;
    for (int i = 0; i < N; ++i) off[i] = deposit_bits(i, group_mask);
    return off;
}

// Runs `op(rb, cb, row_active, col_active)` for every visited block.
template <ExecutionSpace Space, class Op>
void dm_for_each_block(const char* label,
                       std::uint64_t n_qubits,
                       std::uint64_t group_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Op& op) {
    const DmBlockIndexer idx(n_qubits, group_mask, control_mask, control_value_mask);
    if constexpr (Space == ExecutionSpace::Host || Space == ExecutionSpace::HostSerial) {
        // CPU: each work item is one row group × a chunk of up to 2^5 column groups. Column
        // bases inside a chunk are advanced with a carry-skipping increment instead of a full
        // index decode, so the per-block overhead is a few integer ops.
        const std::uint64_t cv = idx.control_value_mask;
        const std::uint64_t row_skip = group_mask | control_mask;
        const std::uint64_t lg = idx.log_groups, la = idx.log_active;
        const std::uint64_t chunk_a = std::min<std::uint64_t>(lg, 5);
        const std::uint64_t chunk_b = std::min<std::uint64_t>(la, 5);
        const std::uint64_t items_a = 1ULL << (la + lg - chunk_a);
        const std::uint64_t rows_b = (1ULL << lg) - (1ULL << la);
        const std::uint64_t items_b = rows_b << (la - chunk_b);
        Kokkos::parallel_for(
            label,
            Kokkos::RangePolicy<SpaceType<Space>>(0, items_a + items_b),
            KOKKOS_LAMBDA(std::uint64_t w) {
                if (w < items_a) {
                    const std::uint64_t per_row = lg - chunk_a;
                    const std::uint64_t rb =
                        insert_zero_at_mask_positions(w >> per_row, row_skip) | cv;
                    std::uint64_t cb = insert_zero_at_mask_positions(
                        (w & ((1ULL << per_row) - 1)) << chunk_a, group_mask);
                    for (std::uint64_t i = 0; i < (1ULL << chunk_a); ++i) {
                        op(rb, cb, true, (cb & control_mask) == cv);
                        cb = ((cb | group_mask) + 1) & ~group_mask;
                    }
                } else {
                    w -= items_a;
                    const std::uint64_t per_row = la - chunk_b;
                    const std::uint64_t r = w >> per_row;
                    const std::uint64_t hi = r / idx.n_inactive_pat;
                    std::uint64_t pat = r - hi * idx.n_inactive_pat;
                    pat += (pat >= idx.active_pattern);
                    const std::uint64_t rb = insert_zero_at_mask_positions(hi, row_skip) |
                                             deposit_bits(pat, control_mask);
                    std::uint64_t cb = insert_zero_at_mask_positions(
                        (w & ((1ULL << per_row) - 1)) << chunk_b, row_skip);
                    for (std::uint64_t i = 0; i < (1ULL << chunk_b); ++i) {
                        op(rb, cb | cv, false, true);
                        cb = ((cb | row_skip) + 1) & ~row_skip;
                    }
                }
            });
    } else {
        // Device: flat index space, innermost dimension = consecutive column groups so that
        // neighbouring threads touch neighbouring addresses (coalesced on LayoutRight).
        Kokkos::parallel_for(
            label,
            Kokkos::RangePolicy<SpaceType<Space>>(0, idx.total),
            KOKKOS_LAMBDA(std::uint64_t g) {
                std::uint64_t rb, cb;
                bool ra, ca;
                idx.decode(g, rb, cb, ra, ca);
                op(rb, cb, ra, ca);
            });
    }
}

// ---------------------------------------------------------------------------------------------
// Block operations
// ---------------------------------------------------------------------------------------------

// General N×N block: B → L B R† where L = U(rb), R = U(cb) come from `MatFn`.
// MatFn::get(base) / get_conj(base) return the (possibly base-dependent) row-major matrix and
// its element-wise conjugate.
template <Precision Prec, ExecutionSpace Space, int N, class MatFn>
struct DmUnitaryBlockOp {
    using C = Complex<Prec>;
    typename DensityMatrix<Prec, Space>::RawView raw;
    Kokkos::Array<std::uint64_t, N> off;
    MatFn mat;

    template <bool Left, bool Right>
    KOKKOS_INLINE_FUNCTION void apply(std::uint64_t rb, std::uint64_t cb) const {
        C b[N][N];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) b[i][j] = raw(rb ^ off[i], cb ^ off[j]);
        if constexpr (Left) {
            const auto& u = mat.get(rb);
            for (int j = 0; j < N; ++j) {
                C t[N];
                for (int i = 0; i < N; ++i) {
                    C s = u[i * N] * b[0][j];
                    for (int k = 1; k < N; ++k) s += u[i * N + k] * b[k][j];
                    t[i] = s;
                }
                for (int i = 0; i < N; ++i) b[i][j] = t[i];
            }
        }
        if constexpr (Right) {
            const auto& uc = mat.get_conj(cb);
            for (int i = 0; i < N; ++i) {
                C t[N];
                for (int j = 0; j < N; ++j) {
                    C s = b[i][0] * uc[j * N];
                    for (int k = 1; k < N; ++k) s += b[i][k] * uc[j * N + k];
                    t[j] = s;
                }
                for (int j = 0; j < N; ++j) b[i][j] = t[j];
            }
        }
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) raw(rb ^ off[i], cb ^ off[j]) = b[i][j];
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t rb,
                                           std::uint64_t cb,
                                           bool ra,
                                           bool ca) const {
        if (ra && ca)
            apply<true, true>(rb, cb);
        else if (ra)
            apply<true, false>(rb, cb);
        else
            apply<false, true>(rb, cb);
    }
};

template <Precision Prec, int N>
struct DmConstantMatrix {
    using Arr = Kokkos::Array<Complex<Prec>, N * N>;
    Arr m, mc;  // matrix and its element-wise conjugate
    explicit DmConstantMatrix(const Arr& m_) : m(m_) {
        for (int k = 0; k < N * N; ++k) mc[k] = conj(m[k]);
    }
    KOKKOS_INLINE_FUNCTION const Arr& get(std::uint64_t) const { return m; }
    KOKKOS_INLINE_FUNCTION const Arr& get_conj(std::uint64_t) const { return mc; }
};

// Diagonal N×N gate: element (i, j) of a block is scaled by
//   (row active ? d[i] : 1) * (col active ? conj(d[j]) : 1).
// Elements whose factor is exactly 1 are not touched at all.
template <Precision Prec, ExecutionSpace Space, int N>
struct DmDiagonalBlockOp {
    using C = Complex<Prec>;
    typename DensityMatrix<Prec, Space>::RawView raw;
    Kokkos::Array<std::uint64_t, N> off;
    // factor[mode][i*N+j], mode: 0 = both, 1 = row only, 2 = col only
    Kokkos::Array<C, 3 * N * N> factor;
    Kokkos::Array<bool, 3 * N * N> trivial;

    DmDiagonalBlockOp(typename DensityMatrix<Prec, Space>::RawView raw_,
                      std::uint64_t group_mask,
                      const Kokkos::Array<C, N>& d)
        : raw(raw_), off(dm_group_offsets<N>(group_mask)) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                factor[0 * N * N + i * N + j] = d[i] * conj(d[j]);
                factor[1 * N * N + i * N + j] = d[i];
                factor[2 * N * N + i * N + j] = conj(d[j]);
            }
        }
        for (int k = 0; k < 3 * N * N; ++k) {
            trivial[k] = factor[k].real() == Float<Prec>{1} && factor[k].imag() == Float<Prec>{0};
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t rb,
                                           std::uint64_t cb,
                                           bool ra,
                                           bool ca) const {
        const int mode = (ra && ca) ? 0 : (ra ? 1 : 2);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const int k = mode * N * N + i * N + j;
                if (!trivial[k]) raw(rb ^ off[i], cb ^ off[j]) *= factor[k];
            }
    }
};

// Permutation of the N group members: |off[i]> → |off[perm[i]]>.
template <Precision Prec, ExecutionSpace Space, int N>
struct DmPermutationBlockOp {
    using C = Complex<Prec>;
    typename DensityMatrix<Prec, Space>::RawView raw;
    Kokkos::Array<std::uint64_t, N> off;
    Kokkos::Array<int, N> src;  // inverse permutation: new[i] = old[src[i]]

    template <bool Left, bool Right>
    KOKKOS_INLINE_FUNCTION void apply(std::uint64_t rb, std::uint64_t cb) const {
        C b[N][N];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const int si = Left ? src[i] : i;
                const int sj = Right ? src[j] : j;
                if (si != i || sj != j) b[i][j] = raw(rb ^ off[si], cb ^ off[sj]);
            }
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const int si = Left ? src[i] : i;
                const int sj = Right ? src[j] : j;
                if (si != i || sj != j) raw(rb ^ off[i], cb ^ off[j]) = b[i][j];
            }
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t rb,
                                           std::uint64_t cb,
                                           bool ra,
                                           bool ca) const {
        if (ra && ca)
            apply<true, true>(rb, cb);
        else if (ra)
            apply<true, false>(rb, cb);
        else
            apply<false, true>(rb, cb);
    }
};

// Superoperator acting on the vectorized N×N block: vec(B) → S vec(B), with vec index i*N+j.
// Used for uncontrolled CPTP maps such as mixtures Σ p_k U_k ρ U_k†. S is stored row-compressed
// because typical channels (Pauli noise, permutation-like gates) are very sparse.
template <Precision Prec, ExecutionSpace Space, int N>
struct DmSuperoperatorBlockOp {
    using C = Complex<Prec>;
    static constexpr int M = N * N;
    typename DensityMatrix<Prec, Space>::RawView raw;
    Kokkos::Array<std::uint64_t, N> off;
    Kokkos::Array<C, M * M> val;
    Kokkos::Array<std::uint8_t, M * M> col;
    Kokkos::Array<std::uint16_t, M + 1> ptr;

    // `dense(y, x)` returns S(y, x).
    template <class Dense>
    DmSuperoperatorBlockOp(typename DensityMatrix<Prec, Space>::RawView raw_,
                           std::uint64_t group_mask,
                           const Dense& dense)
        : raw(raw_), off(dm_group_offsets<N>(group_mask)) {
        int nnz = 0;
        for (int y = 0; y < M; ++y) {
            ptr[y] = static_cast<std::uint16_t>(nnz);
            for (int x = 0; x < M; ++x) {
                const C v = dense(y, x);
                if (v.real() != Float<Prec>{0} || v.imag() != Float<Prec>{0}) {
                    val[nnz] = v;
                    col[nnz] = static_cast<std::uint8_t>(x);
                    ++nnz;
                }
            }
        }
        ptr[M] = static_cast<std::uint16_t>(nnz);
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t rb, std::uint64_t cb, bool, bool) const {
        C v[M];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) v[i * N + j] = raw(rb ^ off[i], cb ^ off[j]);
        for (int y = 0; y < M; ++y) {
            C acc(0);
            for (int e = ptr[y]; e < ptr[y + 1]; ++e) acc += val[e] * v[col[e]];
            raw(rb ^ off[y / N], cb ^ off[y % N]) = acc;
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Convenience entry points
// ---------------------------------------------------------------------------------------------

template <Precision Prec, ExecutionSpace Space, int N>
void dm_apply_constant_unitary(const char* label,
                               std::uint64_t group_mask,
                               std::uint64_t control_mask,
                               std::uint64_t control_value_mask,
                               const Kokkos::Array<Complex<Prec>, N * N>& matrix,
                               DensityMatrix<Prec, Space>& dm) {
    using Op = DmUnitaryBlockOp<Prec, Space, N, DmConstantMatrix<Prec, N>>;
    dm_for_each_block<Space>(
        label,
        dm.n_qubits(),
        group_mask,
        control_mask,
        control_value_mask,
        Op{dm._raw, dm_group_offsets<N>(group_mask), DmConstantMatrix<Prec, N>(matrix)});
}

template <Precision Prec, ExecutionSpace Space, int N>
void dm_apply_diagonal(const char* label,
                       std::uint64_t group_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Kokkos::Array<Complex<Prec>, N>& diag,
                       DensityMatrix<Prec, Space>& dm) {
    dm_for_each_block<Space>(label,
                             dm.n_qubits(),
                             group_mask,
                             control_mask,
                             control_value_mask,
                             DmDiagonalBlockOp<Prec, Space, N>(dm._raw, group_mask, diag));
}

template <Precision Prec, ExecutionSpace Space, int N>
void dm_apply_permutation(const char* label,
                          std::uint64_t group_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          const Kokkos::Array<int, N>& src,
                          DensityMatrix<Prec, Space>& dm) {
    dm_for_each_block<Space>(
        label,
        dm.n_qubits(),
        group_mask,
        control_mask,
        control_value_mask,
        DmPermutationBlockOp<Prec, Space, N>{dm._raw, dm_group_offsets<N>(group_mask), src});
}

}  // namespace scaluq::internal
