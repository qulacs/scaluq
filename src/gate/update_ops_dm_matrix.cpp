// Density-matrix updates for matrix gates, superoperators and measurement projections.
//
// K ≤ 2 targets use register-resident N×N blocks (update_ops_dm_block.hpp).
// K ≥ 3 targets (and mixtures on 3..6 qubits) use Kokkos teams that stage visited blocks in
// team scratch: still a single in-place pass over ρ with no full-size temporaries.

#include "update_ops.hpp"
#include "update_ops_dm_block.hpp"

namespace scaluq::internal {

namespace {
using C = Complex<Prec>;
using RawView = DensityMatrix<Prec, Space>::RawView;
using ExecSpace = SpaceType<Space>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using Member = TeamPolicy::member_type;

// Row-wise access to a dense (LayoutRight) matrix.
struct DenseRows {
    Matrix<Prec, Space> m;
    std::uint64_t n;
    // Σ_k f(row, k) · get(k), with f = m or conj(m).
    template <bool Conj, class Get>
    KOKKOS_INLINE_FUNCTION C row_dot(std::uint64_t row, const Get& get) const {
        C s = 0;
        for (std::uint64_t k = 0; k < n; ++k) s += (Conj ? conj(m(row, k)) : m(row, k)) * get(k);
        return s;
    }
};

// Row-wise access to a CSR matrix: only stored entries are visited.
struct SparseRows {
    SparseMatrix<Prec, Space> m;
    template <bool Conj, class Get>
    KOKKOS_INLINE_FUNCTION C row_dot(std::uint64_t row, const Get& get) const {
        C s = 0;
        for (std::uint32_t idx = m._row_ptr[row]; idx < m._row_ptr[row + 1]; ++idx) {
            const C v = m._vals[idx];
            s += (Conj ? conj(v) : v) * get(m._col_idx[idx]);
        }
        return s;
    }
};

using Scratch1D =
    Kokkos::View<C*, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
using ScratchIdx =
    Kokkos::View<std::uint64_t*, typename ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;

// Team layout for the scratch-staged block kernels.
//
// A team processes `bpt` (a power of two) consecutive blocks. Consecutive block indices are
// consecutive column groups, so with the block index as the fastest-varying work index a warp
// touches contiguous columns of ρ (coalesced), reads the same matrix element (broadcast) and hits
// consecutive scratch words. Scratch element (b, i, j) lives at ((i·N + j) << log_bpt) | b, and
// the per-block bases / activity flags are decoded once per team into scratch.
//
// Host teams are a single thread, so there bpt = 1 (packing would only add scratch).
// Block packing is only used on devices; on the host the packing parameters fold to constants.
constexpr bool kPackBlocks =
    !Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename ExecSpace::memory_space>::accessible;

struct DmTeamLayout {
    std::uint64_t log_bpt;
    std::uint64_t bpt;
    std::uint64_t league;
    int level;
    std::size_t buf_bytes;
    std::size_t total_bytes;

    static std::size_t bytes(std::uint64_t bpt, std::uint64_t nn, int n_bufs) {
        return n_bufs * Scratch1D::shmem_size(bpt * nn) + 3 * ScratchIdx::shmem_size(bpt);
    }

    DmTeamLayout(std::uint64_t nn, std::uint64_t total, int n_bufs) {
        // Host backends cap level-0 scratch at 32 KiB; devices allow a bit more (48 KiB on CUDA).
        constexpr std::size_t level0_budget = (kPackBlocks ? 40 : 32) * 1024;
        log_bpt = 0;
        if constexpr (kPackBlocks) {
            while (log_bpt < 5 && (2ULL << log_bpt) <= total &&
                   bytes(2ULL << log_bpt, nn, n_bufs) <= level0_budget)
                ++log_bpt;
        }
        bpt = 1ULL << log_bpt;
        league = (total + bpt - 1) / bpt;
        buf_bytes = Scratch1D::shmem_size(bpt * nn);
        total_bytes = bytes(bpt, nn, n_bufs);
        level = total_bytes <= level0_budget ? 0 : 1;
    }
};

// Per-team block bases (row, col) and activity flags (bit 0: rows active, bit 1: cols active).
struct DmTeamBlocks {
    ScratchIdx rb, cb, flags;
    std::uint64_t nb;  // blocks actually present in this team (≤ bpt)

    KOKKOS_INLINE_FUNCTION DmTeamBlocks(const Member& team,
                                        int level,
                                        const DmBlockIndexer& idx,
                                        std::uint64_t log_bpt)
        : rb(team.team_scratch(level), 1ULL << log_bpt),
          cb(team.team_scratch(level), 1ULL << log_bpt),
          flags(team.team_scratch(level), 1ULL << log_bpt) {
        const std::uint64_t first = static_cast<std::uint64_t>(team.league_rank()) << log_bpt;
        const std::uint64_t bpt = 1ULL << log_bpt;
        nb = first + bpt <= idx.total ? bpt : idx.total - first;
        const ScratchIdx r = rb, c = cb, f = flags;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nb), [&](std::uint64_t b) {
            std::uint64_t x, y;
            bool ra, ca;
            idx.decode(first + b, x, y, ra, ca);
            r(b) = x;
            c(b) = y;
            f(b) = (ra ? 1 : 0) | (ca ? 2 : 0);
        });
    }
};

inline Kokkos::View<std::uint64_t*, ExecSpace> dm_block_offsets(std::uint64_t target_mask) {
    const std::uint64_t n = 1ULL << std::popcount(target_mask);
    Kokkos::View<std::uint64_t*, ExecSpace> off(
        Kokkos::ViewAllocateWithoutInitializing("dm_block_offsets"), n);
    Kokkos::parallel_for(
        "dm_block_offsets", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(std::uint64_t i) {
            off(i) = deposit_bits(i, target_mask);
        });
    return off;
}

// Fallback for blocks that exceed the backend's per-team scratch limit.  Transform a bounded
// batch of row vectors at a time, then do the same for columns.  The workspace is reused between
// batches, so peak auxiliary memory is independent of the density-matrix size.
template <class Rows>
void dm_apply_large_matrix_batched(std::uint64_t target_mask,
                                   std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   const Rows& rows,
                                   DensityMatrix<Prec, Space>& dm) {
    constexpr std::size_t workspace_budget = 32ULL * 1024 * 1024;
    const std::uint64_t n = 1ULL << std::popcount(target_mask);
    const std::uint64_t dim = dm.dim();
    const std::uint64_t groups = 1ULL
                                 << (dm.n_qubits() - std::popcount(target_mask | control_mask));
    const std::uint64_t n_vectors = groups * dim;
    const std::uint64_t batch_capacity =
        std::max<std::uint64_t>(1, workspace_budget / (sizeof(C) * n));
    Kokkos::View<C**, Kokkos::LayoutRight, ExecSpace> workspace(
        Kokkos::ViewAllocateWithoutInitializing("dm_large_matrix_workspace"), batch_capacity, n);
    const auto off = dm_block_offsets(target_mask);
    const RawView raw = dm._raw;
    const std::uint64_t active_value = control_value_mask & control_mask;
    const std::uint64_t skip = target_mask | control_mask;

    for (std::uint64_t first = 0; first < n_vectors; first += batch_capacity) {
        const std::uint64_t batch = std::min(batch_capacity, n_vectors - first);
        Kokkos::parallel_for(
            "multi_dense_matrix_gate_dm_fallback_gather_left",
            Kokkos::RangePolicy<ExecSpace>(0, batch * n),
            KOKKOS_LAMBDA(std::uint64_t e) {
                const std::uint64_t v = e / n, k = e % n;
                const std::uint64_t g = first + v;
                const std::uint64_t base =
                    insert_zero_at_mask_positions(g / dim, skip) | active_value;
                workspace(v, k) = raw(base ^ off(k), g % dim);
            });
        Kokkos::parallel_for(
            "multi_dense_matrix_gate_dm_fallback_apply_left",
            Kokkos::RangePolicy<ExecSpace>(0, batch * n),
            KOKKOS_LAMBDA(std::uint64_t e) {
                const std::uint64_t v = e / n, i = e % n;
                const std::uint64_t g = first + v;
                const std::uint64_t base =
                    insert_zero_at_mask_positions(g / dim, skip) | active_value;
                raw(base ^ off(i), g % dim) = rows.template row_dot<false>(
                    i, [&](std::uint64_t k) { return workspace(v, k); });
            });
    }

    for (std::uint64_t first = 0; first < n_vectors; first += batch_capacity) {
        const std::uint64_t batch = std::min(batch_capacity, n_vectors - first);
        Kokkos::parallel_for(
            "multi_dense_matrix_gate_dm_fallback_gather_right",
            Kokkos::RangePolicy<ExecSpace>(0, batch * n),
            KOKKOS_LAMBDA(std::uint64_t e) {
                const std::uint64_t v = e / n, k = e % n;
                const std::uint64_t g = first + v;
                const std::uint64_t base =
                    insert_zero_at_mask_positions(g % groups, skip) | active_value;
                workspace(v, k) = raw(g / groups, base ^ off(k));
            });
        Kokkos::parallel_for(
            "multi_dense_matrix_gate_dm_fallback_apply_right",
            Kokkos::RangePolicy<ExecSpace>(0, batch * n),
            KOKKOS_LAMBDA(std::uint64_t e) {
                const std::uint64_t v = e / n, j = e % n;
                const std::uint64_t g = first + v;
                const std::uint64_t base =
                    insert_zero_at_mask_positions(g % groups, skip) | active_value;
                raw(g / groups, base ^ off(j)) = rows.template row_dot<true>(
                    j, [&](std::uint64_t k) { return workspace(v, k); });
            });
    }
    Kokkos::fence();
}

template <class Rows>
void dm_apply_large_matrix(const char* label,
                           std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           const Rows& rows,
                           DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t log_n = std::popcount(target_mask);
    const std::uint64_t n = 1ULL << log_n;
    const std::uint64_t nn = n * n;
    const DmBlockIndexer idx(dm.n_qubits(), target_mask, control_mask, control_value_mask);
    const DmTeamLayout lay(nn, idx.total, 2);
    if (lay.total_bytes > static_cast<std::size_t>(TeamPolicy::scratch_size_max(lay.level))) {
        dm_apply_large_matrix_batched(target_mask, control_mask, control_value_mask, rows, dm);
        return;
    }
    const std::uint64_t lb = lay.log_bpt, nm = n - 1;
    const int level = lay.level;
    const auto off = dm_block_offsets(target_mask);
    const RawView raw = dm._raw;
    Kokkos::parallel_for(
        label,
        TeamPolicy(lay.league, Kokkos::AUTO)
            .set_scratch_size(level, Kokkos::PerTeam(lay.total_bytes)),
        KOKKOS_LAMBDA(const Member& team) {
            Scratch1D cur(team.team_scratch(level), nn << lb);
            Scratch1D nxt(team.team_scratch(level), nn << lb);
            const DmTeamBlocks blk(team, level, idx, lb);
            const std::uint64_t L = kPackBlocks ? lb : 0, bm = (1ULL << L) - 1;
            const std::uint64_t nb = blk.nb, work = nn << L;
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work), [&](std::uint64_t e) {
                const std::uint64_t b = e & bm;
                if (b >= nb) return;
                const std::uint64_t j = (e >> L) & nm, i = e >> (L + log_n);
                cur(e) = raw(blk.rb(b) ^ off(i), blk.cb(b) ^ off(j));
            });
            team.team_barrier();
            // nxt = U · cur  (blocks whose rows are active)
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work), [&](std::uint64_t e) {
                const std::uint64_t b = e & bm;
                if (b >= nb) return;
                const std::uint64_t j = (e >> L) & nm, i = e >> (L + log_n);
                nxt(e) =
                    (blk.flags(b) & 1)
                        ? rows.template row_dot<false>(
                              i,
                              [&](std::uint64_t k) { return cur((((k << log_n) | j) << L) | b); })
                        : cur(e);
            });
            team.team_barrier();
            // ρ block = nxt · U†  (blocks whose columns are active):
            // (nxt U†)(i, j) = Σ_k conj(U(j, k)) nxt(i, k)
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work), [&](std::uint64_t e) {
                const std::uint64_t b = e & bm;
                if (b >= nb) return;
                const std::uint64_t j = (e >> L) & nm, i = e >> (L + log_n);
                raw(blk.rb(b) ^ off(i), blk.cb(b) ^ off(j)) =
                    (blk.flags(b) & 2)
                        ? rows.template row_dot<true>(
                              j,
                              [&](std::uint64_t k) { return nxt((((i << log_n) | k) << L) | b); })
                        : nxt(e);
            });
        });
}

// ρ → c·ρ + Σ_k K_k ρ K_k†, all blocks of the operand qubits (uncontrolled). The block, one
// intermediate product and the accumulator live in team scratch, so ρ is read and written
// exactly once regardless of the number of operators.
void dm_apply_kraus_mixture(std::uint64_t target_mask,
                            Float<Prec> identity_coef,
                            const Kokkos::View<C**, Kokkos::LayoutRight, ExecSpace>& kraus,
                            DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t log_n = std::popcount(target_mask);
    const std::uint64_t n = 1ULL << log_n;
    const std::uint64_t nn = n * n;
    const std::uint64_t r = kraus.extent(0) / n;
    const DmBlockIndexer idx(dm.n_qubits(), target_mask, 0, 0);
    const DmTeamLayout lay(nn, idx.total, 3);
    const std::uint64_t lb = lay.log_bpt, nm = n - 1;
    const int level = lay.level;
    const auto off = dm_block_offsets(target_mask);
    const RawView raw = dm._raw;
    Kokkos::parallel_for(
        "kraus_mixture_gate_dm",
        TeamPolicy(lay.league, Kokkos::AUTO)
            .set_scratch_size(level, Kokkos::PerTeam(lay.total_bytes)),
        KOKKOS_LAMBDA(const Member& team) {
            Scratch1D cur(team.team_scratch(level), nn << lb);
            Scratch1D tmp(team.team_scratch(level), nn << lb);
            Scratch1D acc(team.team_scratch(level), nn << lb);
            const DmTeamBlocks blk(team, level, idx, lb);
            const std::uint64_t L = kPackBlocks ? lb : 0, bm = (1ULL << L) - 1;
            const std::uint64_t nb = blk.nb, work = nn << L;
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work), [&](std::uint64_t e) {
                const std::uint64_t b = e & bm;
                if (b >= nb) return;
                const std::uint64_t j = (e >> L) & nm, i = e >> (L + log_n);
                const C v = raw(blk.rb(b) ^ off(i), blk.cb(b) ^ off(j));
                cur(e) = v;
                acc(e) = v * identity_coef;
            });
            team.team_barrier();
            for (std::uint64_t k = 0; k < r; ++k) {
                const std::uint64_t base = k * n;
                // tmp = K_k · cur
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work), [&](std::uint64_t e) {
                    const std::uint64_t b = e & bm;
                    if (b >= nb) return;
                    const std::uint64_t j = (e >> L) & nm, i = e >> (L + log_n);
                    C s = 0;
                    for (std::uint64_t l = 0; l < n; ++l)
                        s += kraus(base + i, l) * cur((((l << log_n) | j) << L) | b);
                    tmp(e) = s;
                });
                team.team_barrier();
                // acc += tmp · K_k†
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work), [&](std::uint64_t e) {
                    const std::uint64_t b = e & bm;
                    if (b >= nb) return;
                    const std::uint64_t j = (e >> L) & nm, i = e >> (L + log_n);
                    C s = 0;
                    for (std::uint64_t l = 0; l < n; ++l)
                        s += tmp((((i << log_n) | l) << L) | b) * conj(kraus(base + j, l));
                    acc(e) += s;
                });
                team.team_barrier();
            }
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work), [&](std::uint64_t e) {
                const std::uint64_t b = e & bm;
                if (b >= nb) return;
                const std::uint64_t j = (e >> L) & nm, i = e >> (L + log_n);
                raw(blk.rb(b) ^ off(i), blk.cb(b) ^ off(j)) = acc(e);
            });
        });
}

// Functors used inside device lambdas must not be function-local (nvcc extended lambdas).
struct DmMeasurementProjectionOp {
    RawView raw;
    std::uint64_t t;
    int src;  // which diagonal block survives (0 or 1)
    int dst;  // where it is stored
    Float<Prec> scale;
    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t rb, std::uint64_t cb, bool, bool) const {
        const std::uint64_t so = src ? t : 0, d = dst ? t : 0;
        const C v = raw(rb | so, cb | so) * scale;
        raw(rb, cb) = C(0);
        raw(rb, cb | t) = C(0);
        raw(rb | t, cb) = C(0);
        raw(rb | t, cb | t) = C(0);
        raw(rb | d, cb | d) = v;
    }
};
}  // namespace

template <>
void zero_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   Complex<Prec> matrix,
                                   DensityMatrix<Prec, Space>& dm) {
    dm_apply_diagonal<Prec, Space, 1>(
        "zero_target_dense_matrix_gate_dm", 0, control_mask, control_value_mask, {matrix}, dm);
}

template <>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  DensityMatrix<Prec, Space>& dm) {
    Kokkos::Array<C, 16> m;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) m[i * 4 + j] = matrix[i][j];
    dm_apply_constant_unitary<Prec, Space, 4>(
        "two_target_dense_matrix_gate_dm", target_mask, control_mask, control_value_mask, m, dm);
}

template <>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<Prec, Space>& matrix,
                             DensityMatrix<Prec, Space>& dm) {
    dm_apply_large_matrix("multi_dense_matrix_gate_dm",
                          target_mask,
                          control_mask,
                          control_value_mask,
                          DenseRows{matrix, matrix.extent(0)},
                          dm);
}

template <>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& matrix,
                        DensityMatrix<Prec, Space>& dm) {
    dm_apply_large_matrix("sparse_matrix_gate_dm",
                          target_mask,
                          control_mask,
                          control_value_mask,
                          SparseRows{matrix},
                          dm);
}

template <>
void superoperator_gate(std::uint64_t target_mask,
                        const ComplexMatrix& superop,
                        DensityMatrix<Prec, Space>& dm) {
    const int k = std::popcount(target_mask);
    auto dense = [&](int y, int x) {
        return C(static_cast<Float<Prec>>(superop(y, x).real()),
                 static_cast<Float<Prec>>(superop(y, x).imag()));
    };
    if (k == 0) {
        const C s(static_cast<Float<Prec>>(superop(0, 0).real()),
                  static_cast<Float<Prec>>(superop(0, 0).imag()));
        if (s.real() == Float<Prec>{1} && s.imag() == Float<Prec>{0}) return;
        const RawView raw = dm._raw;
        const std::uint64_t n = dm.n_qubits(), mask = dm.dim() - 1;
        Kokkos::parallel_for(
            "superoperator_gate_dm",
            Kokkos::RangePolicy<ExecSpace>(0, dm.dim() * dm.dim()),
            KOKKOS_LAMBDA(std::uint64_t g) { raw(g >> n, g & mask) *= s; });
        return;
    }
    if (k == 1) {
        DmSuperoperatorBlockOp<Prec, Space, 2> op(dm._raw, target_mask, dense);
        dm_for_each_block<Space>("superoperator_gate_dm", dm.n_qubits(), target_mask, 0, 0, op);
    } else if (k == 2) {
        DmSuperoperatorBlockOp<Prec, Space, 4> op(dm._raw, target_mask, dense);
        dm_for_each_block<Space>("superoperator_gate_dm", dm.n_qubits(), target_mask, 0, 0, op);
    } else {
        throw std::runtime_error("superoperator_gate: only up to 2 target qubits are supported.");
    }
}

template <>
void kraus_mixture_gate(std::uint64_t target_mask,
                        double identity_coef,
                        const ComplexMatrix& kraus,
                        DensityMatrix<Prec, Space>& dm) {
    Kokkos::View<C**, Kokkos::LayoutRight, ExecSpace> kv(
        Kokkos::ViewAllocateWithoutInitializing("kraus_mixture_operators"),
        kraus.rows(),
        kraus.cols());
    auto kh = Kokkos::create_mirror_view(kv);
    for (Eigen::Index i = 0; i < kraus.rows(); ++i)
        for (Eigen::Index j = 0; j < kraus.cols(); ++j)
            kh(i, j) = C(static_cast<Float<Prec>>(kraus(i, j).real()),
                         static_cast<Float<Prec>>(kraus(i, j).imag()));
    Kokkos::deep_copy(kv, kh);
    dm_apply_kraus_mixture(target_mask, static_cast<Float<Prec>>(identity_coef), kv, dm);
}

// Projective measurement of the qubit in `target_mask` with outcome `outcome_one`, followed by
// renormalization (1 / probability) and optionally a reset to |0>. One pass over ρ.
template <>
void measurement_projection(std::uint64_t target_mask,
                            bool outcome_one,
                            bool reset,
                            double probability,
                            DensityMatrix<Prec, Space>& dm) {
    const int src = outcome_one ? 1 : 0;
    const int dst = (outcome_one && !reset) ? 1 : 0;
    dm_for_each_block<Space>(
        "measurement_projection_dm",
        dm.n_qubits(),
        target_mask,
        0,
        0,
        DmMeasurementProjectionOp{
            dm._raw, target_mask, src, dst, static_cast<Float<Prec>>(1.0 / probability)});
}

}  // namespace scaluq::internal
