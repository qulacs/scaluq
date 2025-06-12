#include "../prec_space.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   const Matrix<Prec, Space>& matrix,
                                   StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "none_target_dense_matrix_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(0, state.dim() >> std::popcount(control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, control_mask) | control_value_mask;
            state._raw[basis] *= matrix(0, 0);
        });
    Kokkos::fence();
}

template <>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   const Matrix<Prec, Space>& matrix,
                                   StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "none_target_dense_matrix_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, control_mask) | control_value_mask;
            states._raw(batch_id, basis) *= matrix(0, 0);
        });
    Kokkos::fence();
}

template <>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "single_target_dense_matrix_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
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

template <>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "single_target_dense_matrix_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
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

template <>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVector<Prec, Space>& state) {
    std::uint64_t target_bit_right = -target_mask & target_mask;
    std::uint64_t target_bit_left = target_mask ^ target_bit_right;

    Kokkos::parallel_for(
        "double_target_dense_matrix_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
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

template <>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVectorBatched<Prec, Space>& states) {
    std::uint64_t target_bit_right = -target_mask & target_mask;
    std::uint64_t target_bit_left = target_mask ^ target_bit_right;

    Kokkos::parallel_for(
        "double_target_dense_matrix_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
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

template <>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    std::uint64_t control_value_mask,
                                    const Matrix<Prec, Space>& matrix,
                                    StateVector<Prec, Space>& state) {
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);
    Kokkos::View<Complex<Prec>*, SpaceType<Space>> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
    Kokkos::parallel_for(
        "multi_target_dense_matrix_gate (initialize)",
        Kokkos::RangePolicy<SpaceType<Space>>(0, state.dim()),
        KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update(i) = -1;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::parallel_for(
        "multi_target_dense_matrix_gate (update)",
        Kokkos::TeamPolicy<SpaceType<Space>>(
            SpaceType<Space>(),
            state.dim() >> std::popcount(target_mask | control_mask),
            Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
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

template <>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    std::uint64_t control_value_mask,
                                    const Matrix<Prec, Space>& matrix,
                                    StateVectorBatched<Prec, Space>& states) {
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);

    Kokkos::View<Complex<Prec>**, Kokkos::LayoutRight, SpaceType<Space>> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), states.batch_size(), states.dim());

    Kokkos::parallel_for(
        "multi_target_dense_matrix_gate (initialize)",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update(batch_id, i) = 0;
            } else {
                update(batch_id, i) = states._raw(batch_id, i);  // 制御条件を満たさないインデクス
            }
        });
    Kokkos::fence();

    std::uint64_t outer_size = states.dim() >> std::popcount(target_mask | control_mask);
    std::uint64_t outer_mask = ~target_mask & ((1ULL << states.n_qubits()) - 1);
    Kokkos::parallel_for(
        "multi_target_dense_matrix_gate (update)",
        Kokkos::TeamPolicy<SpaceType<Space>>(
            SpaceType<Space>(), outer_size * states.batch_size(), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t basis = team.league_rank() % outer_size;
            std::uint64_t batch_id = team.league_rank() / outer_size;
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
                std::uint64_t dst_index =
                    internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex<Prec> sum = Float<Prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t c, Complex<Prec>& inner_sum) {
                        std::uint64_t src_index =
                            internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += matrix(r, c) * states._raw(batch_id, src_index);
                    },
                    sum);
                update(batch_id, dst_index) = sum;
            });
            team.team_barrier();
        });
    Kokkos::fence();

    states._raw = update;
}

template <>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVector<Prec, Space>& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    if (target_qubit_index_count == 0) {
        none_target_dense_matrix_gate<Prec>(control_mask, control_value_mask, matrix, state);
    } else if (target_qubit_index_count == 1) {
        single_target_dense_matrix_gate<Prec>(
            target_mask, control_mask, control_value_mask, matrix, state);
    } else if (target_qubit_index_count == 2) {
        double_target_dense_matrix_gate<Prec>(
            target_mask, control_mask, control_value_mask, matrix, state);
    } else {
        multi_target_dense_matrix_gate<Prec>(
            target_mask, control_mask, control_value_mask, matrix, state);
    }
}

template <>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVectorBatched<Prec, Space>& states) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    if (target_qubit_index_count == 0) {
        none_target_dense_matrix_gate<Prec>(control_mask, control_value_mask, matrix, states);
    } else if (target_qubit_index_count == 1) {
        single_target_dense_matrix_gate<Prec>(
            target_mask, control_mask, control_value_mask, matrix, states);
    } else if (target_qubit_index_count == 2) {
        double_target_dense_matrix_gate<Prec>(
            target_mask, control_mask, control_value_mask, matrix, states);
    } else {
        multi_target_dense_matrix_gate<Prec>(
            target_mask, control_mask, control_value_mask, matrix, states);
    }
}
}  // namespace scaluq::internal
