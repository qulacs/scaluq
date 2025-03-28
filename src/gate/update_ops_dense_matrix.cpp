#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec, ExecutionSpace Space>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Prec, Space>& matrix,
                                   StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<SpaceType<Space>>(0, state.dim() >> std::popcount(control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis = insert_zero_at_mask_positions(it, control_mask) | control_mask;
            state._raw[basis] *= matrix(0, 0);
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                  \
    template void none_target_dense_matrix_gate( \
        std::uint64_t, const Matrix<Prec, Space>&, StateVector<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Prec, Space>& matrix,
                                   StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis = insert_zero_at_mask_positions(it, control_mask) | control_mask;
            states._raw(batch_id, basis) *= matrix(0, 0);
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                  \
    template void none_target_dense_matrix_gate( \
        std::uint64_t, const Matrix<Prec, Space>&, StateVectorBatched<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<Prec> val0 = state._raw[basis_0];
            Complex<Prec> val1 = state._raw[basis_1];
            Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                 \
    template void one_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix2x2<Prec>&, StateVector<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<Prec> val0 = states._raw(batch_id, basis_0);
            Complex<Prec> val1 = states._raw(batch_id, basis_1);
            Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            states._raw(batch_id, basis_0) = res0;
            states._raw(batch_id, basis_1) = res1;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                 \
    template void one_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix2x2<Prec>&, StateVectorBatched<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVector<Prec, Space>& state) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
            Complex<Prec> val0 = state._raw[basis_0];
            Complex<Prec> val1 = state._raw[basis_1];
            Complex<Prec> val2 = state._raw[basis_2];
            Complex<Prec> val3 = state._raw[basis_3];
            Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1 + matrix[0][2] * val2 +
                                 matrix[0][3] * val3;
            Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1 + matrix[1][2] * val2 +
                                 matrix[1][3] * val3;
            Complex<Prec> res2 = matrix[2][0] * val0 + matrix[2][1] * val1 + matrix[2][2] * val2 +
                                 matrix[2][3] * val3;
            Complex<Prec> res3 = matrix[3][0] * val0 + matrix[3][1] * val1 + matrix[3][2] * val2 +
                                 matrix[3][3] * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                 \
    template void two_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix4x4<Prec>&, StateVector<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVectorBatched<Prec, Space>& states) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_0 | target_mask;
            Complex<Prec> val0 = states._raw(batch_id, basis_0);
            Complex<Prec> val1 = states._raw(batch_id, basis_1);
            Complex<Prec> val2 = states._raw(batch_id, basis_2);
            Complex<Prec> val3 = states._raw(batch_id, basis_3);
            Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1 + matrix[0][2] * val2 +
                                 matrix[0][3] * val3;
            Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1 + matrix[1][2] * val2 +
                                 matrix[1][3] * val3;
            Complex<Prec> res2 = matrix[2][0] * val0 + matrix[2][1] * val1 + matrix[2][2] * val2 +
                                 matrix[2][3] * val3;
            Complex<Prec> res3 = matrix[3][0] * val0 + matrix[3][1] * val1 + matrix[3][2] * val2 +
                                 matrix[3][3] * val3;
            states._raw(batch_id, basis_0) = res0;
            states._raw(batch_id, basis_1) = res1;
            states._raw(batch_id, basis_2) = res2;
            states._raw(batch_id, basis_3) = res3;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                 \
    template void two_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix4x4<Prec>&, StateVectorBatched<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<Prec> val0 = state._raw[basis_0];
            Complex<Prec> val1 = state._raw[basis_1];
            Complex<Prec> res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1;
            Complex<Prec> res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                    \
    template void single_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&, StateVector<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<Prec> val0 = states._raw(batch_id, basis_0);
            Complex<Prec> val1 = states._raw(batch_id, basis_1);
            Complex<Prec> res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1;
            Complex<Prec> res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1;
            states._raw(batch_id, basis_0) = res0;
            states._raw(batch_id, basis_1) = res1;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                                               \
    template void single_target_dense_matrix_gate(std::uint64_t,              \
                                                  std::uint64_t,              \
                                                  const Matrix<Prec, Space>&, \
                                                  StateVectorBatched<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVector<Prec, Space>& state) {
    std::uint64_t target_bit_right = -target_mask & target_mask;
    std::uint64_t target_bit_left = target_mask ^ target_bit_right;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_bit_right;
            std::uint64_t basis_2 = basis_0 | target_bit_left;
            std::uint64_t basis_3 = basis_0 | target_mask;
            Complex<Prec> val0 = state._raw[basis_0];
            Complex<Prec> val1 = state._raw[basis_1];
            Complex<Prec> val2 = state._raw[basis_2];
            Complex<Prec> val3 = state._raw[basis_3];
            Complex<Prec> res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1 + matrix(0, 2) * val2 +
                                 matrix(0, 3) * val3;
            Complex<Prec> res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1 + matrix(1, 2) * val2 +
                                 matrix(1, 3) * val3;
            Complex<Prec> res2 = matrix(2, 0) * val0 + matrix(2, 1) * val1 + matrix(2, 2) * val2 +
                                 matrix(2, 3) * val3;
            Complex<Prec> res3 = matrix(3, 0) * val0 + matrix(3, 1) * val1 + matrix(3, 2) * val2 +
                                 matrix(3, 3) * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                    \
    template void double_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&, StateVector<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVectorBatched<Prec, Space>& states) {
    std::uint64_t target_bit_right = -target_mask & target_mask;
    std::uint64_t target_bit_left = target_mask ^ target_bit_right;

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_bit_right;
            std::uint64_t basis_2 = basis_0 | target_bit_left;
            std::uint64_t basis_3 = basis_0 | target_mask;
            Complex<Prec> val0 = states._raw(batch_id, basis_0);
            Complex<Prec> val1 = states._raw(batch_id, basis_1);
            Complex<Prec> val2 = states._raw(batch_id, basis_2);
            Complex<Prec> val3 = states._raw(batch_id, basis_3);
            Complex<Prec> res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1 + matrix(0, 2) * val2 +
                                 matrix(0, 3) * val3;
            Complex<Prec> res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1 + matrix(1, 2) * val2 +
                                 matrix(1, 3) * val3;
            Complex<Prec> res2 = matrix(2, 0) * val0 + matrix(2, 1) * val1 + matrix(2, 2) * val2 +
                                 matrix(2, 3) * val3;
            Complex<Prec> res3 = matrix(3, 0) * val0 + matrix(3, 1) * val1 + matrix(3, 2) * val2 +
                                 matrix(3, 3) * val3;
            states._raw(batch_id, basis_0) = res0;
            states._raw(batch_id, basis_1) = res1;
            states._raw(batch_id, basis_2) = res2;
            states._raw(batch_id, basis_3) = res3;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec, Space)                                               \
    template void double_target_dense_matrix_gate(std::uint64_t,              \
                                                  std::uint64_t,              \
                                                  const Matrix<Prec, Space>&, \
                                                  StateVectorBatched<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Prec, Space>& matrix,
                                    StateVector<Prec, Space>& state) {
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);

    Kokkos::View<Complex<Prec>*, SpaceType<Space>> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
    Kokkos::parallel_for(
        Kokkos::RangePolicy<SpaceType<Space>>(0, state.dim()), KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<SpaceType<Space>>(
            SpaceType<Space>(),
            state.dim() >> std::popcount(target_mask | control_mask),
            Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) | control_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
                std::uint64_t dst_index =
                    internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex<Prec> sum = Float<Prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t c, Complex<Prec>& inner_sum) {
                        std::uint64_t src_index =
                            internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += matrix(r, c) * state._raw(src_index);
                    },
                    sum);
                update(dst_index) = sum;
            });
            team.team_barrier();
        });
    Kokkos::fence();

    state._raw = update;
}
#define FUNC_MACRO(Prec, Space)                   \
    template void multi_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&, StateVector<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Prec, Space>& matrix,
                                    StateVectorBatched<Prec, Space>& states) {
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);
    const std::uint64_t outer_mask = ~target_mask & ((1ULL << states.n_qubits()) - 1);

    Kokkos::View<Complex<Prec>**, Kokkos::LayoutRight, SpaceType<Space>> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), states.batch_size(), states.dim());

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(batch_id, i) = 0;
            } else {
                update(batch_id, i) = states._raw(batch_id, i);
            }
        });
    Kokkos::fence();

    Kokkos::View<Complex<Prec>**,
                 Kokkos::LayoutRight,
                 SpaceType<Space>,
                 Kokkos::MemoryTraits<Kokkos::Atomic>>
        update_atomic(update);

    const std::uint64_t outer_dim = states.dim() >> std::popcount(target_mask | control_mask);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<4>>(
            {0, 0, 0, 0}, {states.batch_size(), outer_dim, matrix_dim, matrix_dim}),
        KOKKOS_LAMBDA(const std::uint64_t batch_id,
                      const std::uint64_t outer_idx,
                      const std::uint64_t r,
                      const std::uint64_t c) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(outer_idx, target_mask | control_mask) | control_mask;

            std::uint64_t dst_index =
                internal::insert_zero_at_mask_positions(r, outer_mask) | basis;

            std::uint64_t src_index =
                internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            update_atomic(batch_id, dst_index) += matrix(r, c) * states._raw(batch_id, src_index);
        });
    Kokkos::fence();

    states._raw = update;
}
#define FUNC_MACRO(Prec, Space)                                              \
    template void multi_target_dense_matrix_gate(std::uint64_t,              \
                                                 std::uint64_t,              \
                                                 const Matrix<Prec, Space>&, \
                                                 StateVectorBatched<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVector<Prec, Space>& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    if (target_qubit_index_count == 0) {
        none_target_dense_matrix_gate<Prec>(control_mask, matrix, state);
    } else if (target_qubit_index_count == 1) {
        single_target_dense_matrix_gate<Prec>(target_mask, control_mask, matrix, state);
    } else if (target_qubit_index_count == 2) {
        double_target_dense_matrix_gate<Prec>(target_mask, control_mask, matrix, state);
    } else {
        multi_target_dense_matrix_gate<Prec>(target_mask, control_mask, matrix, state);
    }
}
#define FUNC_MACRO(Prec, Space)      \
    template void dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&, StateVector<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec, ExecutionSpace Space>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVectorBatched<Prec, Space>& states) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    if (target_qubit_index_count == 0) {
        none_target_dense_matrix_gate<Prec>(control_mask, matrix, states);
    } else if (target_qubit_index_count == 1) {
        single_target_dense_matrix_gate<Prec>(target_mask, control_mask, matrix, states);
    } else if (target_qubit_index_count == 2) {
        double_target_dense_matrix_gate<Prec>(target_mask, control_mask, matrix, states);
    } else {
        multi_target_dense_matrix_gate<Prec>(target_mask, control_mask, matrix, states);
    }
}
#define FUNC_MACRO(Prec, Space)                                 \
    template void dense_matrix_gate(std::uint64_t,              \
                                    std::uint64_t,              \
                                    const Matrix<Prec, Space>&, \
                                    StateVectorBatched<Prec, Space>&);
SCALUQ_CALL_MACRO_FOR_PRECISION_AND_EXECUTION_SPACE(FUNC_MACRO)
#undef FUNC_MACRO
}  // namespace scaluq::internal
