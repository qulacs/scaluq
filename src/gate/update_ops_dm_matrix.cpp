// Density-matrix updates for multi-qubit matrix gates.
//
// K ≤ 2 targets use register-resident N×N blocks (update_ops_dm_block.hpp).
// K ≥ 3 targets use Kokkos teams that stage visited blocks in team scratch: still a single
// in-place pass over ρ with no full-size temporaries.

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

}  // namespace scaluq::internal
