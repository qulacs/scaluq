#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const matrix_2_2& matrix,
                                  StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex res0 = matrix.val[0][0] * val0 + matrix.val[0][1] * val1;
            Complex res1 = matrix.val[1][0] * val0 + matrix.val[1][1] * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}

void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const matrix_4_4& matrix,
                                  StateVector& state) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex val2 = state._raw[basis_2];
            Complex val3 = state._raw[basis_3];
            Complex res0 = matrix.val[0][0] * val0 + matrix.val[0][1] * val1 +
                           matrix.val[0][2] * val2 + matrix.val[0][3] * val3;
            Complex res1 = matrix.val[1][0] * val0 + matrix.val[1][1] * val1 +
                           matrix.val[1][2] * val2 + matrix.val[1][3] * val3;
            Complex res2 = matrix.val[2][0] * val0 + matrix.val[2][1] * val1 +
                           matrix.val[2][2] * val2 + matrix.val[2][3] * val3;
            Complex res3 = matrix.val[3][0] * val0 + matrix.val[3][1] * val1 +
                           matrix.val[3][2] * val2 + matrix.val[3][3] * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}

void single_qubit_dense_matrix_gate_view(std::uint64_t target_mask,
                                         const Matrix& matrix,
                                         StateVector& state) {
    const std::uint64_t loop_dim = state.dim() >> 1;
    const std::uint64_t mask_low = target_mask - 1;
    const std::uint64_t mask_high = ~mask_low;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const std::uint64_t state_index) {
            std::uint64_t basis_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            std::uint64_t basis_1 = basis_0 + target_mask;
            Complex v0 = state._raw[basis_0];
            Complex v1 = state._raw[basis_1];
            state._raw[basis_0] = matrix(0, 0) * v0 + matrix(0, 1) * v1;
            state._raw[basis_1] = matrix(1, 0) * v0 + matrix(1, 1) * v1;
        });
    Kokkos::fence();
}

void double_qubit_dense_matrix_gate(std::uint64_t target_mask,
                                    const Matrix& matrix,
                                    StateVector& state) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;

    const std::uint64_t loop_dim = state.dim() >> 2;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 = insert_zero_at_mask_positions(it, target_mask);
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex val2 = state._raw[basis_2];
            Complex val3 = state._raw[basis_3];
            Complex res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1 + matrix(0, 2) * val2 +
                           matrix(0, 3) * val3;
            Complex res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1 + matrix(1, 2) * val2 +
                           matrix(1, 3) * val3;
            Complex res2 = matrix(2, 0) * val0 + matrix(2, 1) * val1 + matrix(2, 2) * val2 +
                           matrix(2, 3) * val3;
            Complex res3 = matrix(3, 0) * val0 + matrix(3, 1) * val1 + matrix(3, 2) * val2 +
                           matrix(3, 3) * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}

void single_qubit_control_single_qubit_dense_matrix_gate(std::uint64_t target_mask,
                                                         std::uint64_t control_mask,
                                                         const Matrix& matrix,
                                                         StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 2, KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1;
            Complex res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}

void single_qubit_control_multi_qubit_dense_matrix_gate(std::uint64_t target_mask,
                                                        std::uint64_t control_mask,
                                                        const Matrix& matrix,
                                                        StateVector& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    const std::uint64_t matrix_dim = 1ULL << target_qubit_index_count;
    const std::uint64_t insert_index_count = target_qubit_index_count + 1;
    const std::uint64_t loop_dim = state.dim() >> insert_index_count;

    std::vector<std::uint64_t> matrix_mask_list = create_matrix_mask_list(target_mask);
    Kokkos::View<std::uint64_t*> matrix_mask_view =
        convert_host_vector_to_device_view(matrix_mask_list);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            std::uint64_t basis_0 = team.league_rank();
            basis_0 =
                insert_zero_at_mask_positions(basis_0, control_mask | target_mask) | control_mask;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, matrix_dim), [&](const std::uint64_t y) {
                    Complex sum = 0;
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team, matrix_dim),
                        [&](const std::uint64_t x, Complex& inner_sum) {
                            inner_sum += matrix(y, x) * state._raw[basis_0 ^ matrix_mask_view(x)];
                        },
                        sum);
                    buffer[y] = sum;
                });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim),
                                 [=](const std::uint64_t y) {
                                     state._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
                                 });
        });
    Kokkos::fence();
}

void multi_qubit_control_single_qubit_dense_matrix_gate(std::uint64_t target_mask,
                                                        std::uint64_t control_mask,
                                                        const Matrix& matrix,
                                                        StateVector& state) {
    std::uint64_t control_qubit_index_count = std::popcount(control_mask);
    if (control_qubit_index_count == 1) {
        single_qubit_control_single_qubit_dense_matrix_gate(
            target_mask, control_mask, matrix, state);
        return;
    }

    const std::uint64_t loop_dim = state.dim() >> (control_qubit_index_count + 1);
    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1;
            Complex res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}

void multi_qubit_control_multi_qubit_dense_matrix_gate(std::uint64_t target_mask,
                                                       std::uint64_t control_mask,
                                                       const Matrix& matrix,
                                                       StateVector& state) {
    const std::uint64_t control_qubit_index_count = std::popcount(control_mask);
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    const std::uint64_t matrix_dim = 1ULL << target_qubit_index_count;
    const std::uint64_t insert_index_count = target_qubit_index_count + control_qubit_index_count;
    const std::uint64_t loop_dim = state.dim() >> insert_index_count;
    std::vector<std::uint64_t> matrix_mask_list = create_matrix_mask_list(target_mask);
    Kokkos::View<std::uint64_t*> matrix_mask_view =
        convert_host_vector_to_device_view(matrix_mask_list);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            std::uint64_t basis_0 = team.league_rank();
            basis_0 =
                insert_zero_at_mask_positions(basis_0, target_mask | control_mask) | control_mask;
            basis_0 ^= control_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t y) {
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t x, Complex& inner_sum) {
                        inner_sum += matrix(y, x) * state._raw[basis_0 ^ matrix_mask_view(x)];
                    },
                    sum);
                buffer[y] = sum;
            });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim),
                                 [=](const std::uint64_t y) {
                                     state._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
                                 });
        });
    Kokkos::fence();
}

void multi_qubit_dense_matrix_gate_parallel(std::uint64_t target_mask,
                                            const Matrix& matrix,
                                            StateVector& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    const std::uint64_t matrix_dim = 1ULL << target_qubit_index_count;
    const std::uint64_t loop_dim = state.dim() >> target_qubit_index_count;
    std::vector<std::uint64_t> matrix_mask_list = create_matrix_mask_list(target_mask);
    Kokkos::View<std::uint64_t*> matrix_mask_view =
        convert_host_vector_to_device_view(matrix_mask_list);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            std::uint64_t basis_0 = team.league_rank();
            basis_0 = insert_zero_at_mask_positions(basis_0, target_mask);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t y) {
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t x, Complex& inner_sum) {
                        inner_sum += matrix(y, x) * state._raw[basis_0 ^ matrix_mask_view(x)];
                    },
                    sum);
                buffer[y] = sum;
            });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim),
                                 [=](const std::uint64_t y) {
                                     state._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
                                 });
        });
    Kokkos::fence();
}

void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix& matrix,
                       StateVector& state) {
    if (std::popcount(target_mask) == 1) {
        if (std::popcount(control_mask) == 0) {
            single_qubit_dense_matrix_gate_view(target_mask, matrix, state);
        } else if (std::popcount(control_mask) == 1) {
            single_qubit_control_single_qubit_dense_matrix_gate(
                target_mask, control_mask, matrix, state);
        } else {
            multi_qubit_control_single_qubit_dense_matrix_gate(
                target_mask, control_mask, matrix, state);
        }
    } else {
        if (std::popcount(control_mask) == 0) {
            multi_qubit_dense_matrix_gate_parallel(target_mask, matrix, state);
        } else if (std::popcount(control_mask) == 1) {
            single_qubit_control_multi_qubit_dense_matrix_gate(
                target_mask, control_mask, matrix, state);
        } else {
            // multiple control qubit
            multi_qubit_control_multi_qubit_dense_matrix_gate(
                target_mask, control_mask, matrix, state);
        }
    }
}
}  // namespace internal
}  // namespace scaluq
