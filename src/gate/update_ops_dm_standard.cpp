// Density-matrix updates for standard / Pauli gates.
//
// Every gate is applied in a single in-place pass over the touched elements of ρ using the
// block machinery in update_ops_dm_block.hpp (see the comment there for the block scheme).

#include "update_ops.hpp"
#include "update_ops_dm_block.hpp"

namespace scaluq::internal {

namespace {
using C = Complex<Prec>;
using RawView = DensityMatrix<Prec, Space>::RawView;

KOKKOS_INLINE_FUNCTION bool odd_parity(std::uint64_t x) { return Kokkos::popcount(x) & 1; }

// Scales row r by f(r) and column c by conj(f(c)), where f(r) = parity(r & pfm) ? f1 : f0.
// Used for diagonal Pauli strings (Z/I only) and their rotations.
struct DmParityDiagonalOp {
    RawView raw;
    std::uint64_t phase_flip_mask;
    C f0, f1;

    KOKKOS_INLINE_FUNCTION C f(std::uint64_t x) const {
        return odd_parity(x & phase_flip_mask) ? f1 : f0;
    }
    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t r,
                                           std::uint64_t c,
                                           bool ra,
                                           bool ca) const {
        C factor = ra ? f(r) : C(1);
        if (ca) factor *= conj(f(c));
        raw(r, c) *= factor;
    }
};

// 2×2 matrix on the pair {base, base ^ bit_flip_mask} of a Pauli string.
//   Pauli:          [[0, φ(b0)], [φ(b1), 0]] · coef
//   PauliRotation:  [[cos, i sin φ(b0)], [i sin φ(b1), cos]]
// where φ(x) = ±global_phase depending on parity(x & phase_flip_mask).
struct DmPauliPairMatrix {
    std::uint64_t bit_flip_mask;
    std::uint64_t phase_flip_mask;
    C diag;     // diagonal entry
    C offdiag;  // off-diagonal scale (global phase folded in)

    KOKKOS_INLINE_FUNCTION Kokkos::Array<C, 4> get(std::uint64_t base) const {
        const C p0 = odd_parity(base & phase_flip_mask) ? -offdiag : offdiag;
        const C p1 = odd_parity((base ^ bit_flip_mask) & phase_flip_mask) ? -offdiag : offdiag;
        return {diag, p0, p1, diag};
    }
    KOKKOS_INLINE_FUNCTION Kokkos::Array<C, 4> get_conj(std::uint64_t base) const {
        auto u = get(base);
        for (int k = 0; k < 4; ++k) u[k] = conj(u[k]);
        return u;
    }
};

// Pauli string with bit flips on the pair {base, base ^ bit_flip_mask}:
//   U = [[0, φ0], [φ1, 0]], so (U B U†)_{ij} = φ_i conj(φ'_j) B_{1-i, 1-j}: a pure permutation
//   with phases, no general 2×2 products needed.
struct DmPauliBlockOp {
    RawView raw;
    std::uint64_t bit_flip_mask;
    std::uint64_t phase_flip_mask;
    C scale;  // global phase × coef

    KOKKOS_INLINE_FUNCTION void phases(std::uint64_t base, C& p0, C& p1) const {
        p0 = odd_parity(base & phase_flip_mask) ? -scale : scale;
        p1 = odd_parity((base ^ bit_flip_mask) & phase_flip_mask) ? -scale : scale;
    }
    KOKKOS_INLINE_FUNCTION void operator()(std::uint64_t rb,
                                           std::uint64_t cb,
                                           bool ra,
                                           bool ca) const {
        const std::uint64_t r1 = rb ^ bit_flip_mask, c1 = cb ^ bit_flip_mask;
        C pr0(1), pr1(1), pc0(1), pc1(1);
        if (ra) phases(rb, pr0, pr1);
        if (ca) {
            phases(cb, pc0, pc1);
            pc0 = conj(pc0);
            pc1 = conj(pc1);
        }
        if (ra && ca) {
            const C v00 = raw(rb, cb), v01 = raw(rb, c1), v10 = raw(r1, cb), v11 = raw(r1, c1);
            raw(rb, cb) = pr0 * pc0 * v11;
            raw(rb, c1) = pr0 * pc1 * v10;
            raw(r1, cb) = pr1 * pc0 * v01;
            raw(r1, c1) = pr1 * pc1 * v00;
        } else if (ra) {
            const C v00 = raw(rb, cb), v01 = raw(rb, c1), v10 = raw(r1, cb), v11 = raw(r1, c1);
            raw(rb, cb) = pr0 * v10;
            raw(rb, c1) = pr0 * v11;
            raw(r1, cb) = pr1 * v00;
            raw(r1, c1) = pr1 * v01;
        } else {
            const C v00 = raw(rb, cb), v01 = raw(rb, c1), v10 = raw(r1, cb), v11 = raw(r1, c1);
            raw(rb, cb) = v01 * pc0;
            raw(rb, c1) = v00 * pc1;
            raw(r1, cb) = v11 * pc0;
            raw(r1, c1) = v10 * pc1;
        }
    }
};

void dm_apply_pauli_pair(const char* label,
                         std::uint64_t control_mask,
                         std::uint64_t control_value_mask,
                         std::uint64_t bit_flip_mask,
                         std::uint64_t phase_flip_mask,
                         C diag,
                         C offdiag,
                         DensityMatrix<Prec, Space>& dm) {
    const std::uint64_t pivot_mask = 1ULL << (Kokkos::bit_width(bit_flip_mask) - 1);
    using Op = DmUnitaryBlockOp<Prec, Space, 2, DmPauliPairMatrix>;
    dm_for_each_block<Space>(
        label,
        dm.n_qubits(),
        pivot_mask,
        control_mask,
        control_value_mask,
        Op{dm._raw, {0, bit_flip_mask}, {bit_flip_mask, phase_flip_mask, diag, offdiag}});
}

Kokkos::Array<C, 4> to_array(const Matrix2x2<Prec>& m) {
    return {m[0][0], m[0][1], m[1][0], m[1][1]};
}
}  // namespace

template <>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t control_value_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 DensityMatrix<Prec, Space>& dm) {
    if (bit_flip_mask == 0) {
        if (control_mask == 0 && phase_flip_mask == 0 &&
            (coef * conj(coef)).real() == Float<Prec>{1})
            return;
        dm_for_each_block<Space>("apply_pauli_dm",
                                 dm.n_qubits(),
                                 0,
                                 control_mask,
                                 control_value_mask,
                                 DmParityDiagonalOp{dm._raw, phase_flip_mask, coef, -coef});
        return;
    }
    const C global_phase = PHASE_M90ROT<Prec>()[std::popcount(bit_flip_mask & phase_flip_mask) % 4];
    dm_for_each_block<Space>(
        "apply_pauli_dm",
        dm.n_qubits(),
        1ULL << (Kokkos::bit_width(bit_flip_mask) - 1),
        control_mask,
        control_value_mask,
        DmPauliBlockOp{dm._raw, bit_flip_mask, phase_flip_mask, global_phase * coef});
}

template <>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          DensityMatrix<Prec, Space>& dm) {
    const C half_angle = angle * coef / Float<Prec>{2};
    const C cosval = internal::cos(-half_angle);
    const C sinval = internal::sin(-half_angle);
    const C imag_sin = C(0, 1) * sinval;
    if (bit_flip_mask == 0) {
        dm_for_each_block<Space>(
            "apply_pauli_rotation_dm",
            dm.n_qubits(),
            0,
            control_mask,
            control_value_mask,
            DmParityDiagonalOp{dm._raw, phase_flip_mask, cosval + imag_sin, cosval - imag_sin});
        return;
    }
    const C global_phase = PHASE_M90ROT<Prec>()[std::popcount(bit_flip_mask & phase_flip_mask) % 4];
    dm_apply_pauli_pair("apply_pauli_rotation_dm",
                        control_mask,
                        control_value_mask,
                        bit_flip_mask,
                        phase_flip_mask,
                        cosval,
                        imag_sin * global_phase,
                        dm);
}

template <>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  DensityMatrix<Prec, Space>& dm) {
    dm_apply_constant_unitary<Prec, Space, 2>("one_target_dense_matrix_gate_dm",
                                              target_mask,
                                              control_mask,
                                              control_value_mask,
                                              to_array(matrix),
                                              dm);
}

template <>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     DensityMatrix<Prec, Space>& dm) {
    dm_apply_diagonal<Prec, Space, 2>("one_target_diagonal_matrix_gate_dm",
                                      target_mask,
                                      control_mask,
                                      control_value_mask,
                                      {diag[0], diag[1]},
                                      dm);
}

template <>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           DensityMatrix<Prec, Space>& dm) {
    dm_apply_diagonal<Prec, Space, 2>("one_target_phase_gate_dm",
                                      target_mask,
                                      control_mask,
                                      control_value_mask,
                                      {C(1), phase},
                                      dm);
}

// Uncontrolled global phase is a no-op on ρ; controlled one is a 0-target diagonal gate.
template <>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       DensityMatrix<Prec, Space>& dm) {
    if (control_mask == 0) return;
    dm_apply_diagonal<Prec, Space, 1>("global_phase_gate_dm",
                                      0,
                                      control_mask,
                                      control_value_mask,
                                      {internal::polar<Prec>(Float<Prec>{1}, angle)},
                                      dm);
}

template <>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    dm_apply_permutation<Prec, Space, 2>(
        "x_gate_dm", target_mask, control_mask, control_value_mask, {1, 0}, dm);
}

template <>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    dm_apply_constant_unitary<Prec, Space, 2>("y_gate_dm",
                                              target_mask,
                                              control_mask,
                                              control_value_mask,
                                              {C(0), C(0, -1), C(0, 1), C(0)},
                                              dm);
}

template <>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm) {
    dm_apply_diagonal<Prec, Space, 2>(
        "z_gate_dm", target_mask, control_mask, control_value_mask, {C(1), C(-1)}, dm);
}

template <>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    dm_apply_constant_unitary<Prec, Space, 2>("rx_gate_dm",
                                              target_mask,
                                              control_mask,
                                              control_value_mask,
                                              {C(cosval), C(0, -sinval), C(0, -sinval), C(cosval)},
                                              dm);
}

template <>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    dm_apply_constant_unitary<Prec, Space, 2>("ry_gate_dm",
                                              target_mask,
                                              control_mask,
                                              control_value_mask,
                                              {C(cosval), C(-sinval), C(sinval), C(cosval)},
                                              dm);
}

template <>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    dm_apply_diagonal<Prec, Space, 2>("rz_gate_dm",
                                      target_mask,
                                      control_mask,
                                      control_value_mask,
                                      {C(cosval, -sinval), C(cosval, sinval)},
                                      dm);
}

template <>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               DensityMatrix<Prec, Space>& dm) {
    dm_apply_permutation<Prec, Space, 4>(
        "swap_gate_dm", target_mask, control_mask, control_value_mask, {0, 2, 1, 3}, dm);
}

// A sequence of k swaps would cost k passes; instead gather ρ'(r, c) = ρ(σ⁻¹(r), σ⁻¹(c)) in
// one pass into a fresh buffer, where σ⁻¹ applies the swaps in reverse order.
template <>
void permutation_gate(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& swap_schedule,
                      DensityMatrix<Prec, Space>& dm) {
    if (swap_schedule.empty()) return;
    if (swap_schedule.size() == 1) {
        const auto& [a, b] = swap_schedule.front();
        swap_gate((1ULL << a) | (1ULL << b), 0, 0, dm);
        return;
    }
    const std::uint64_t dim = dm.dim();
    const std::uint64_t n_qubits = dm.n_qubits();
    const std::uint64_t n_swaps = swap_schedule.size();
    Kokkos::View<std::uint64_t*, Kokkos::HostSpace> schedule_host("swap_schedule", 2 * n_swaps);
    for (std::uint64_t i = 0; i < n_swaps; ++i) {
        schedule_host(2 * i) = swap_schedule[i].first;
        schedule_host(2 * i + 1) = swap_schedule[i].second;
    }
    auto schedule = Kokkos::create_mirror_view_and_copy(SpaceType<Space>(), schedule_host);
    Kokkos::View<std::uint64_t*, SpaceType<Space>> inverse(
        Kokkos::ViewAllocateWithoutInitializing("permutation_inverse"), dim);
    Kokkos::parallel_for(
        "permutation_gate_dm_table",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dim),
        KOKKOS_LAMBDA(std::uint64_t r) {
            std::uint64_t x = r;
            for (std::uint64_t i = n_swaps; i-- > 0;) {
                const std::uint64_t a = schedule(2 * i), b = schedule(2 * i + 1);
                const std::uint64_t diff = ((x >> a) ^ (x >> b)) & 1;
                x ^= (diff << a) | (diff << b);
            }
            inverse(r) = x;
        });
    RawView updated(Kokkos::ViewAllocateWithoutInitializing("state"), dim, dim);
    const RawView src = dm._raw;
    Kokkos::parallel_for(
        "permutation_gate_dm",
        Kokkos::RangePolicy<SpaceType<Space>>(0, dim * dim),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t r = g >> n_qubits, c = g & (dim - 1);
            updated(r, c) = src(inverse(r), inverse(c));
        });
    dm._raw = updated;
}

// ECR in the basis {0, pc, pt, pc|pt} (pc: physical control, pt: physical target).
template <>
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              DensityMatrix<Prec, Space>& dm) {
    const Float<Prec> s = static_cast<Float<Prec>>(INVERSE_SQRT2());
    const Kokkos::Array<C, 16> ecr = {C(0),
                                      C(s),
                                      C(0),
                                      C(0, s),  //
                                      C(s),
                                      C(0),
                                      C(0, -s),
                                      C(0),  //
                                      C(0),
                                      C(0, s),
                                      C(0),
                                      C(s),  //
                                      C(0, -s),
                                      C(0),
                                      C(s),
                                      C(0)};
    using Op = DmUnitaryBlockOp<Prec, Space, 4, DmConstantMatrix<Prec, 4>>;
    dm_for_each_block<Space>("ecr_gate_dm",
                             dm.n_qubits(),
                             physical_target_mask | physical_control_mask,
                             control_mask,
                             control_value_mask,
                             Op{dm._raw,
                                {0,
                                 physical_control_mask,
                                 physical_target_mask,
                                 physical_control_mask | physical_target_mask},
                                DmConstantMatrix<Prec, 4>(ecr)});
}

}  // namespace scaluq::internal
