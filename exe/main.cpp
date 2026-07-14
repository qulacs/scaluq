#include <chrono>
#include <cstdio>
#include <scaluq/all.hpp>

using namespace scaluq;

// Benchmark: two-pass vs single-pass 4-block for controlled DM gates on CPU.
//
// For a controlled-U gate: ρ → U_ctrl ρ U_ctrl†.
//
// Two-pass: left (apply U to ctrl-active rows × ALL cols), right (apply U† to ALL rows × ctrl-active cols).
//   Total: 2 × n_pairs × dim iters.  Some (ctrl-active row × ctrl-active col) elements touched twice.
//
// 4-block (treat ctrl+tgt as joint 2-qubit space):
//   n_quads² blocks, each updating the full 4×4 (ctrl,tgt) submatrix.
//   CRITICAL: base = insert_zero(it, ctrl|tgt) with NO control_value_mask forcing.
//   r[0]=ctrl=0,tgt=0; r[1]=ctrl=0,tgt=1; r[2]=ctrl=1,tgt=0; r[3]=ctrl=1,tgt=1.
//   Each element is touched exactly once.  Total: n_quads² × k_nontrivial iters.

static constexpr auto P = Precision::F64;
static constexpr auto S = ExecutionSpace::Host;
using C = internal::Complex<P>;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ─── Two-pass: X gate ─────────────────────────────────────────────────────
void two_pass_x(std::uint64_t tgt_mask,
                std::uint64_t ctrl_mask,
                std::uint64_t ctrl_val_mask,
                DensityMatrix<P, S>& dm) {
    const std::uint64_t n = dm.dim() >> std::popcount(tgt_mask | ctrl_mask);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<internal::SpaceType<S>>(0, n * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim(), col = g % dm.dim();
            std::uint64_t r0 = internal::insert_zero_at_mask_positions(it, ctrl_mask | tgt_mask)
                               | ctrl_val_mask;
            Kokkos::kokkos_swap(dm._raw(r0, col), dm._raw(r0 | tgt_mask, col));
        });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<internal::SpaceType<S>>(0, dm.dim() * n),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n, it = g % n;
            std::uint64_t c0 = internal::insert_zero_at_mask_positions(it, ctrl_mask | tgt_mask)
                               | ctrl_val_mask;
            Kokkos::kokkos_swap(dm._raw(row, c0), dm._raw(row, c0 | tgt_mask));
        });
}

// ─── 4-block: CX — 6 swaps per block, each element touched once ───────────
// ρ'_{ij} = ρ_{CX(i),CX(j)}.  For ctrl-active=ctrl=1:
//   In each 4×4 (ctrl,tgt) block:
//     ctrl=0 row × ctrl=1 col: col-swap   (r[0..1], c[2..3]) — 2 swaps
//     ctrl=1 row × ctrl=0 col: row-swap   (r[2..3], c[0..1]) — 2 swaps
//     ctrl=1 row × ctrl=1 col: diag-swap  (r[2..3], c[2..3]) — 2 swaps
//     ctrl=0 row × ctrl=0 col: unchanged
// CORRECT: base does NOT have ctrl forced — all 4 (ctrl,tgt) combinations.
void four_block_cx(std::uint64_t tgt_mask,
                   std::uint64_t ctrl_mask,
                   DensityMatrix<P, S>& dm) {
    const std::uint64_t n = dm.dim() >> std::popcount(tgt_mask | ctrl_mask);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<internal::SpaceType<S>>(0, n * n),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it_r = g / n, it_c = g % n;
            // NO | ctrl_val: generate all 4 (ctrl,tgt) combos
            std::uint64_t br = internal::insert_zero_at_mask_positions(it_r, ctrl_mask | tgt_mask);
            std::uint64_t bc = internal::insert_zero_at_mask_positions(it_c, ctrl_mask | tgt_mask);
            std::uint64_t r0=br, r1=br|tgt_mask, r2=br|ctrl_mask, r3=r2|tgt_mask;
            std::uint64_t c0=bc, c1=bc|tgt_mask, c2=bc|ctrl_mask, c3=c2|tgt_mask;
            Kokkos::kokkos_swap(dm._raw(r0, c2), dm._raw(r0, c3));  // ctrl=0 row, ctrl=1 col
            Kokkos::kokkos_swap(dm._raw(r1, c2), dm._raw(r1, c3));
            Kokkos::kokkos_swap(dm._raw(r2, c0), dm._raw(r3, c0));  // ctrl=1 row, ctrl=0 col
            Kokkos::kokkos_swap(dm._raw(r2, c1), dm._raw(r3, c1));
            Kokkos::kokkos_swap(dm._raw(r2, c2), dm._raw(r3, c3));  // ctrl=1 row, ctrl=1 col
            Kokkos::kokkos_swap(dm._raw(r2, c3), dm._raw(r3, c2));
        });
}

// ─── Two-pass: diagonal controlled gate ───────────────────────────────────
void two_pass_diag(std::uint64_t tgt_mask,
                   std::uint64_t ctrl_mask,
                   std::uint64_t ctrl_val_mask,
                   const internal::DiagonalMatrix2x2<P>& diag,
                   DensityMatrix<P, S>& dm) {
    const std::uint64_t n = dm.dim() >> std::popcount(tgt_mask | ctrl_mask);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<internal::SpaceType<S>>(0, n * dm.dim()),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it = g / dm.dim(), col = g % dm.dim();
            std::uint64_t r0 = internal::insert_zero_at_mask_positions(it, ctrl_mask | tgt_mask)
                               | ctrl_val_mask;
            dm._raw(r0, col) *= diag[0];
            dm._raw(r0 | tgt_mask, col) *= diag[1];
        });
    Kokkos::parallel_for(
        Kokkos::RangePolicy<internal::SpaceType<S>>(0, dm.dim() * n),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t row = g / n, it = g % n;
            std::uint64_t c0 = internal::insert_zero_at_mask_positions(it, ctrl_mask | tgt_mask)
                               | ctrl_val_mask;
            dm._raw(row, c0) *= internal::conj(diag[0]);
            dm._raw(row, c0 | tgt_mask) *= internal::conj(diag[1]);
        });
}

// ─── 4-block: diagonal controlled gate ────────────────────────────────────
// G_diag = [1, 1, diag[0], diag[1]] in {ctrl=0,tgt=0/1; ctrl=1,tgt=0/1} basis.
// factor[yr*4+yc] = G_diag[yr] * conj(G_diag[yc]).
// Elements in ctrl=0 row × ctrl=0 col block always have factor=1 (no-op).
// Up to 12 non-trivial elements per block (4 always no-op).
void four_block_diag(std::uint64_t tgt_mask,
                     std::uint64_t ctrl_mask,
                     const internal::DiagonalMatrix2x2<P>& diag,
                     DensityMatrix<P, S>& dm) {
    // G_diag for ctrl=1 active: G[0]=1, G[1]=1, G[2]=diag[0], G[3]=diag[1]
    Kokkos::Array<C, 4> gd = {C(1), C(1), diag[0], diag[1]};
    Kokkos::Array<C, 16> fac;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            fac[y*4+x] = gd[y] * internal::conj(gd[x]);

    const std::uint64_t n = dm.dim() >> std::popcount(tgt_mask | ctrl_mask);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<internal::SpaceType<S>>(0, n * n),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it_r = g / n, it_c = g % n;
            std::uint64_t br = internal::insert_zero_at_mask_positions(it_r, ctrl_mask | tgt_mask);
            std::uint64_t bc = internal::insert_zero_at_mask_positions(it_c, ctrl_mask | tgt_mask);
            std::uint64_t r[4] = {br, br|tgt_mask, br|ctrl_mask, br|ctrl_mask|tgt_mask};
            std::uint64_t c[4] = {bc, bc|tgt_mask, bc|ctrl_mask, bc|ctrl_mask|tgt_mask};
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                    if (fac[y*4+x].real() != 1 || fac[y*4+x].imag() != 0)
                        dm._raw(r[y], c[x]) *= fac[y*4+x];
        });
}

// ─── 4-block: CZ optimized (avoid checking factor==1 at runtime) ──────────
// CZ: G_diag=[1,1,1,-1]; only 6 elements have factor=-1.
void four_block_cz(std::uint64_t tgt_mask,
                   std::uint64_t ctrl_mask,
                   DensityMatrix<P, S>& dm) {
    const std::uint64_t n = dm.dim() >> std::popcount(tgt_mask | ctrl_mask);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<internal::SpaceType<S>>(0, n * n),
        KOKKOS_LAMBDA(std::uint64_t g) {
            std::uint64_t it_r = g / n, it_c = g % n;
            std::uint64_t br = internal::insert_zero_at_mask_positions(it_r, ctrl_mask | tgt_mask);
            std::uint64_t bc = internal::insert_zero_at_mask_positions(it_c, ctrl_mask | tgt_mask);
            std::uint64_t r0=br, r1=br|tgt_mask, r2=br|ctrl_mask, r3=r2|tgt_mask;
            std::uint64_t c0=bc, c1=bc|tgt_mask, c2=bc|ctrl_mask, c3=c2|tgt_mask;
            // Negate where factor(row)*conj(factor(col)) = (-1)*1 or 1*(-1)
            // factor(k) = -1 iff (k_ctrl=1 AND k_tgt=1)
            dm._raw(r0, c3) = -dm._raw(r0, c3);
            dm._raw(r1, c3) = -dm._raw(r1, c3);
            dm._raw(r2, c3) = -dm._raw(r2, c3);
            dm._raw(r3, c0) = -dm._raw(r3, c0);
            dm._raw(r3, c1) = -dm._raw(r3, c1);
            dm._raw(r3, c2) = -dm._raw(r3, c2);
            // (r3, c3): factor = (-1)*(-1) = 1 → no-op
        });
}

// ─── Correctness check ─────────────────────────────────────────────────────
void check_correctness(int n) {
    std::uint64_t ctrl_mask = 1ULL << 1, tgt_mask = 1ULL << 0, ctrl_val = ctrl_mask;
    internal::DiagonalMatrix2x2<P> z_diag = {C(1), C(-1)};

    // CX correctness
    {
        auto dm_ref = DensityMatrix<P, S>::Haar_random_state(n);
        auto dm_tst = dm_ref;
        two_pass_x(tgt_mask, ctrl_mask, ctrl_val, dm_ref);
        four_block_cx(tgt_mask, ctrl_mask, dm_tst);
        auto m1 = dm_ref.get_matrix(), m2 = dm_tst.get_matrix();
        double err = 0;
        for (int i = 0; i < (int)m1.rows(); ++i)
            for (int j = 0; j < (int)m1.cols(); ++j)
                err = std::max(err, std::abs(m1(i,j) - m2(i,j)));
        std::printf("  CX correctness check (n=%d): max_err = %e %s\n",
                    n, err, err < 1e-12 ? "PASS" : "FAIL");
    }

    // CZ correctness
    {
        auto dm_ref = DensityMatrix<P, S>::Haar_random_state(n);
        auto dm_tst = dm_ref;
        two_pass_diag(tgt_mask, ctrl_mask, ctrl_val, z_diag, dm_ref);
        four_block_cz(tgt_mask, ctrl_mask, dm_tst);
        auto m1 = dm_ref.get_matrix(), m2 = dm_tst.get_matrix();
        double err = 0;
        for (int i = 0; i < (int)m1.rows(); ++i)
            for (int j = 0; j < (int)m1.cols(); ++j)
                err = std::max(err, std::abs(m1(i,j) - m2(i,j)));
        std::printf("  CZ correctness check (n=%d): max_err = %e %s\n",
                    n, err, err < 1e-12 ? "PASS" : "FAIL");
    }
}

template <typename F>
double bench(const char* name, int n_qubits, int warmup, int reps, F fn) {
    auto dm0 = DensityMatrix<P, S>::Haar_random_state(n_qubits);
    for (int i = 0; i < warmup; ++i) { auto dm = dm0; fn(dm); }
    Kokkos::fence();
    double t0 = now_ms();
    for (int i = 0; i < reps; ++i) { auto dm = dm0; fn(dm); }
    Kokkos::fence();
    double ms = (now_ms() - t0) / reps;
    std::printf("    %-50s %7.2f ms\n", name, ms);
    return ms;
}

int main() {
    scaluq::initialize();
    std::printf("=== Correctness check ===\n");
    for (int n : {4, 6}) check_correctness(n);

    std::printf("\n=== Performance benchmark ===\n");
    std::printf("two-pass:  2 parallel_for × n_pairs×dim iters\n");
    std::printf("4-block:   1 parallel_for × n_quads² iters, each touching 16 elements\n\n");

    for (int n : {8, 10, 12}) {
        std::uint64_t dim = 1ULL << n;
        std::printf("n=%d  dim=%llu (DM = %llu MB)\n",
                    n, dim, dim*dim*16/(1024*1024));
        std::uint64_t ctrl_mask = 1ULL << 1, tgt_mask = 1ULL << 0, ctrl_val = ctrl_mask;
        internal::DiagonalMatrix2x2<P> z_diag = {C(1), C(-1)};
        int warmup = 3, reps = 10;

        std::printf("  [CX gate]\n");
        double t1 = bench("two-pass (2 kernels)", n, warmup, reps,
            [&](auto& dm){ two_pass_x(tgt_mask, ctrl_mask, ctrl_val, dm); });
        double t2 = bench("4-block  (6 swaps/block, 1 kernel, correct base)", n, warmup, reps,
            [&](auto& dm){ four_block_cx(tgt_mask, ctrl_mask, dm); });
        std::printf("    4-block speedup: %.2fx\n\n", t1/t2);

        std::printf("  [CZ gate]\n");
        double t3 = bench("two-pass (2 kernels)", n, warmup, reps,
            [&](auto& dm){ two_pass_diag(tgt_mask, ctrl_mask, ctrl_val, z_diag, dm); });
        double t4 = bench("4-block  (generic, factor check)", n, warmup, reps,
            [&](auto& dm){ four_block_diag(tgt_mask, ctrl_mask, z_diag, dm); });
        double t5 = bench("4-block  (CZ-specialized, 6 negations)", n, warmup, reps,
            [&](auto& dm){ four_block_cz(tgt_mask, ctrl_mask, dm); });
        std::printf("    4-block-generic speedup: %.2fx\n", t3/t4);
        std::printf("    4-block-cz speedup:      %.2fx\n\n", t3/t5);
    }

    scaluq::finalize();
}
