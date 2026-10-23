#include <algorithm>
#include <bit>
#include <cstdint>
#include <utility>

#include "update_ops.hpp"

namespace scaluq::internal {

namespace {
// Lane targets are the least significant matrix-index bits. XOR each subset
// into the input lanes and select the corresponding matrix entry per output lane.
template <std::uint64_t LaneTargets, std::size_t Xor, UpdatableStateVector State>
KOKKOS_INLINE_FUNCTION auto dense_simd_accumulate(const Matrix<State::prec, State::space>& matrix,
                                                  std::uint64_t r,
                                                  std::uint64_t c,
                                                  const SimdComplex<State::prec>& input,
                                                  const SimdComplex<State::prec>& sum) {
    using Simd = SimdComplex<State::prec>;
    using Coef = typename Simd::Coef;
    constexpr auto count = std::popcount(LaneTargets);
    [[maybe_unused]] constexpr auto bit0 = std::countr_zero(LaneTargets);
    constexpr auto rest = LaneTargets & (LaneTargets - 1);
    [[maybe_unused]] constexpr auto bit1 = std::countr_zero(rest);
    [[maybe_unused]] constexpr auto bit2 = std::countr_zero(rest & (rest - 1));
    const auto coef = [&] {
        if constexpr (count == 0) {
            return Coef::splat(matrix(r, c));
        } else if constexpr (count == 1) {
            return Coef::template select_complex_lane_bit<bit0>(matrix(r, c + Xor),
                                                                matrix(r + 1, c + (1 ^ Xor)));
        } else if constexpr (count == 2) {
            return Coef::template select_complex_lane_bits<bit0, bit1>(
                matrix(r, c + Xor),
                matrix(r + 1, c + (1 ^ Xor)),
                matrix(r + 2, c + (2 ^ Xor)),
                matrix(r + 3, c + (3 ^ Xor)));
        } else {
            return Coef::template select_complex_lane_bits<bit0, bit1, bit2>(
                matrix(r, c + Xor),
                matrix(r + 1, c + (1 ^ Xor)),
                matrix(r + 2, c + (2 ^ Xor)),
                matrix(r + 3, c + (3 ^ Xor)),
                matrix(r + 4, c + (4 ^ Xor)),
                matrix(r + 5, c + (5 ^ Xor)),
                matrix(r + 6, c + (6 ^ Xor)),
                matrix(r + 7, c + (7 ^ Xor)));
        }
    }();
    const auto value = [&] {
        if constexpr (Xor == 0)
            return input;
        else {
            constexpr std::uint64_t lane_xor = [] {
                std::uint64_t result = 0, remaining = LaneTargets;
                for (std::size_t bit = 0; remaining; ++bit) {
                    if (Xor & (1ULL << bit)) result |= remaining & -remaining;
                    remaining &= remaining - 1;
                }
                return result;
            }();
            return input.template permute_complex_lanes_xor<lane_xor>();
        }
    }();
    const auto next = coef.fma(value, sum);
    if constexpr (Xor + 1 < (1ULL << count))
        return dense_simd_accumulate<LaneTargets, Xor + 1, State>(matrix, r, c, input, next);
    else
        return next;
}

template <typename Simd, std::size_t... Rows>
KOKKOS_INLINE_FUNCTION auto dense_simd_zeros(const Simd& zero, std::index_sequence<Rows...>) {
    return Kokkos::Array<Simd, sizeof...(Rows)>{(static_cast<void>(Rows), zero)...};
}

template <std::uint64_t LaneTargets, std::size_t RowBlock, UpdatableStateVector State>
void multi_dense_matrix_gate_simd(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix<State::prec, State::space>& matrix,
                                  State& state) {
    using ExecSpace = SpaceType<State::space>;
    using Simd = SimdComplex<State::prec>;
    using Policy = Kokkos::TeamPolicy<ExecSpace>;
    constexpr auto lanes = Simd::complex_lanes;
    constexpr auto count = std::popcount(LaneTargets);
    constexpr auto inner_dim = 1ULL << count;
    const auto outer_targets = target_mask ^ LaneTargets;
    const auto skip_mask = outer_targets | control_mask;
    const auto outer_dim = 1ULL << std::popcount(outer_targets);
    if constexpr (RowBlock > 1) {
        if (outer_dim < RowBlock) {
            multi_dense_matrix_gate_simd<LaneTargets, 1>(
                target_mask, control_mask, control_value_mask, matrix, state);
            return;
        }
    }
    const auto blocks = (state.flat_dim() >> std::popcount(skip_mask)) / lanes;
    const auto outer_mask = ~outer_targets & (state.dim() - 1);
    const int team_size = blocks == 1 ? ExecSpace().concurrency() : 1;
    // Kokkos scratch views only guarantee eight-byte alignment by default.
    constexpr auto alignment = sizeof(Complex<State::prec>) * lanes;
    const auto scratch_bytes = alignment * outer_dim;
    const int scratch_level =
        scratch_bytes + alignment <= static_cast<std::size_t>(Policy::scratch_size_max(0)) ? 0 : 1;
    const auto policy =
        Policy(blocks, team_size)
            .set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes + alignment));
    constexpr const char* label = count == 0   ? "multi_dense_matrix_gate_simd"
                                  : count == 1 ? "multi_dense_matrix_gate_simd_middle1"
                                  : count == 2 ? "multi_dense_matrix_gate_simd_middle2"
                                               : "multi_dense_matrix_gate_simd_middle3";
    Kokkos::parallel_for(
        label, policy, KOKKOS_LAMBDA(const typename Policy::member_type& team) {
            const auto basis =
                insert_zero_at_mask_positions(team.league_rank() * lanes, skip_mask) |
                control_value_mask;
            auto* input = static_cast<Complex<State::prec>*>(
                team.team_scratch(scratch_level).get_shmem_aligned(scratch_bytes, alignment));
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, outer_dim), [&](std::uint64_t c) {
                const auto index = basis | insert_zero_at_mask_positions(c, outer_mask);
                Simd::load_aligned(&state.at_unsafe(index)).store_aligned(input + c * lanes);
            });
            team.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, outer_dim / RowBlock), [&](std::uint64_t row_block) {
                    using Scalar = std::conditional_t<State::prec == Precision::F64, double, float>;
                    const auto zero = Simd(simd_ops::Simd<Scalar>(Scalar{0}));
                    auto sum0 = dense_simd_zeros(zero, std::make_index_sequence<RowBlock>{});
                    auto sum1 = sum0;
                    const auto r = row_block * RowBlock;
                    std::uint64_t c = 0;
                    for (; c + 1 < outer_dim; c += 2) {
                        const auto v0 = Simd::load_aligned(input + c * lanes);
                        const auto v1 = Simd::load_aligned(input + (c + 1) * lanes);
                        for (std::size_t row = 0; row < RowBlock; ++row) {
                            sum0[row] = dense_simd_accumulate<LaneTargets, 0, State>(
                                matrix, (r + row) * inner_dim, c * inner_dim, v0, sum0[row]);
                            sum1[row] = dense_simd_accumulate<LaneTargets, 0, State>(
                                matrix, (r + row) * inner_dim, (c + 1) * inner_dim, v1, sum1[row]);
                        }
                    }
                    if (c < outer_dim) {
                        const auto value = Simd::load_aligned(input + c * lanes);
                        for (std::size_t row = 0; row < RowBlock; ++row) {
                            sum0[row] = dense_simd_accumulate<LaneTargets, 0, State>(
                                matrix, (r + row) * inner_dim, c * inner_dim, value, sum0[row]);
                        }
                    }
                    for (std::size_t row = 0; row < RowBlock; ++row) {
                        const auto index =
                            basis | insert_zero_at_mask_positions(r + row, outer_mask);
                        (sum0[row] + sum1[row]).store_aligned(&state.at_unsafe(index));
                    }
                });
            team.team_barrier();
        });
}

// Instantiate only the masks supported by the native vector width.
template <std::uint64_t LaneTargets = 0, UpdatableStateVector State>
void dispatch_dense_matrix_simd(std::uint64_t target_mask,
                                std::uint64_t control_mask,
                                std::uint64_t control_value_mask,
                                const Matrix<State::prec, State::space>& matrix,
                                State& state) {
    constexpr auto lanes = SimdComplex<State::prec>::complex_lanes;
    if ((target_mask & (lanes - 1)) == LaneTargets) {
        // Lane selection increases register pressure. Use four broadcast rows
        // and two middle1 rows; keep middle2/middle3 unblocked.
        constexpr std::size_t row_block = LaneTargets == 0                  ? 4
                                          : std::popcount(LaneTargets) == 1 ? 2
                                                                            : 1;
        multi_dense_matrix_gate_simd<LaneTargets, row_block>(
            target_mask, control_mask, control_value_mask, matrix, state);
    } else if constexpr (LaneTargets + 1 < lanes) {
        dispatch_dense_matrix_simd<LaneTargets + 1>(
            target_mask, control_mask, control_value_mask, matrix, state);
    }
}

template <UpdatableStateVector State>
void multi_dense_matrix_gate_scalar(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    std::uint64_t control_value_mask,
                                    const Matrix<State::prec, State::space>& matrix,
                                    State& state) {
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);
    using Policy = Kokkos::TeamPolicy<ExecSpace>;
    using Scratch = Kokkos::View<ComplexType*,
                                 typename ExecSpace::scratch_memory_space,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const std::uint64_t skip_mask = target_mask | control_mask;
    const std::uint64_t blocks = state.flat_dim() >> std::popcount(skip_mask);
    const std::uint64_t outer_mask = ~target_mask & (state.dim() - 1);
    if (blocks == 0) return;
    const auto scratch_bytes = Scratch::shmem_size(matrix_dim);
    const int scratch_level =
        scratch_bytes <= static_cast<std::size_t>(Policy::scratch_size_max(0)) ? 0 : 1;
    auto policy = Policy(blocks, Kokkos::AUTO)
                      .set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes));
    const auto kernel = KOKKOS_LAMBDA(const typename Policy::member_type& team) {
        const std::uint64_t basis =
            insert_zero_at_mask_positions(team.league_rank(), skip_mask) | control_value_mask;
        const Scratch input(team.team_scratch(scratch_level), matrix_dim);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t c) {
            input(c) = state.at_unsafe(basis | insert_zero_at_mask_positions(c, outer_mask));
        });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
            // Independent accumulators shorten the dependency chain and
            // let the compiler vectorize the contiguous complex products.
            Float<State::prec> real[4]{}, imag[4]{};
            std::uint64_t c = 0;
            for (; c + 3 < matrix_dim; c += 4) {
                for (int k = 0; k < 4; ++k) {
                    const auto& a = matrix(r, c + k);
                    const auto& b = input(c + k);
                    real[k] += a.real() * b.real() - a.imag() * b.imag();
                    imag[k] += a.real() * b.imag() + a.imag() * b.real();
                }
            }
            ComplexType sum((real[0] + real[1]) + (real[2] + real[3]),
                            (imag[0] + imag[1]) + (imag[2] + imag[3]));
            for (; c < matrix_dim; ++c) sum += matrix(r, c) * input(c);
            state.at_unsafe(basis | insert_zero_at_mask_positions(r, outer_mask)) = sum;
        });
        // Protect scratch reuse when a team proceeds to its next block.
        team.team_barrier();
    };
    // Respect backend occupancy recommendations while distributing workers
    // across independent blocks and capping the team at the number of rows.
    const auto workers = std::max<std::uint64_t>(1, ExecSpace().concurrency() / blocks);
    const auto recommended = policy.team_size_recommended(kernel, Kokkos::ParallelForTag{});
    const auto capacity = policy.team_size_max(kernel, Kokkos::ParallelForTag{});
    const int team_size = std::min({matrix_dim,
                                    static_cast<std::uint64_t>(capacity),
                                    std::max<std::uint64_t>(workers, recommended)});
    policy =
        Policy(blocks, team_size).set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes));
    Kokkos::parallel_for("multi_dense_matrix_gate_scalar", policy, kernel);
}

}  // namespace

template <UpdatableStateVector State>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<State::prec, State::space>& matrix,
                             State& state) {
    if constexpr (supports_gate_simd<State>) {
        if (can_use_gate_simd(control_mask, state)) {
            dispatch_dense_matrix_simd(
                target_mask, control_mask, control_value_mask, matrix, state);
            return;
        }
    }
    multi_dense_matrix_gate_scalar(target_mask, control_mask, control_value_mask, matrix, state);
}

template <UpdatableStateVector State>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<State::prec, State::space>& mat,
                        State& state) {
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    State update;
    if constexpr (std::same_as<State, StateVector<State::prec, State::space>>) {
        update = State::uninitialized_state(state.n_qubits());
    } else {
        update = State::uninitialized_state(state.batch_size(), state.n_qubits());
    }

    Kokkos::parallel_for(
        "sparse_matrix_gate (initialize)",
        Kokkos::RangePolicy<ExecSpace>(0, state.flat_dim()),
        KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update.at_unsafe(i) = 0;
            } else {
                update.at_unsafe(i) = state.at_unsafe(i);
            }
        });

    const std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::parallel_for(
        "sparse_matrix_gate (update)",
        Kokkos::TeamPolicy<ExecSpace>(ExecSpace(),
                                      state.flat_dim() >> std::popcount(target_mask | control_mask),
                                      Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mat._rows), [&](std::uint64_t r) {
                const std::uint64_t dst_index =
                    insert_zero_at_mask_positions(r, outer_mask) | basis;
                ComplexType sum = Float<State::prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, mat._row_ptr[r], mat._row_ptr[r + 1]),
                    [&](std::uint64_t idx, ComplexType& inner_sum) {
                        const std::uint64_t c = mat._col_idx[idx];
                        const std::uint64_t src_index =
                            insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += mat._vals[idx] * state.at_unsafe(src_index);
                    },
                    sum);
                update.at_unsafe(dst_index) = sum;
            });
            team.team_barrier();
        });

    std::swap(state._raw, update._raw);
}

#define INSTANTIATE_FLAT_STATE_OVERLOADS(Func, ...)                                       \
    template void Func<StateVector<Prec, Space>>(__VA_ARGS__, StateVector<Prec, Space>&); \
    template void Func<StateVectorBatched<Prec, Space>>(__VA_ARGS__,                      \
                                                        StateVectorBatched<Prec, Space>&)
// clang-format off
INSTANTIATE_FLAT_STATE_OVERLOADS(multi_dense_matrix_gate,       std::uint64_t, std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(sparse_matrix_gate,            std::uint64_t, std::uint64_t, std::uint64_t, const SparseMatrix<Prec, Space>&);

#undef INSTANTIATE_FLAT_STATE_OVERLOADS
// clang-format on

}  // namespace scaluq::internal
