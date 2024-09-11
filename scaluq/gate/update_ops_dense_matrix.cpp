#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2& matrix,
                                  StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}

void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4& matrix,
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
            Complex res0 = matrix[0][0] * val0 + matrix[0][1] * val1 + matrix[0][2] * val2 +
                           matrix[0][3] * val3;
            Complex res1 = matrix[1][0] * val0 + matrix[1][1] * val1 + matrix[1][2] * val2 +
                           matrix[1][3] * val3;
            Complex res2 = matrix[2][0] * val0 + matrix[2][1] * val1 + matrix[2][2] * val2 +
                           matrix[2][3] * val3;
            Complex res3 = matrix[3][0] * val0 + matrix[3][1] * val1 + matrix[3][2] * val2 +
                           matrix[3][3] * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}

void multi_control_single_target_dense_matrix_gate(std::uint64_t target_mask,
                                                   std::uint64_t control_mask,
                                                   const Matrix& matrix,
                                                   StateVector& state) {
    std::uint64_t control_qubit_index_count = std::popcount(control_mask);
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

void multi_control_double_target_dense_matrix_gate(std::uint64_t target_mask,
                                                   std::uint64_t control_mask,
                                                   const Matrix& matrix,
                                                   StateVector& state) {
    const std::uint64_t insert_index_count = 2 + std::popcount(control_mask);
    const std::uint64_t loop_dim = state.dim() >> insert_index_count;
    const std::uint64_t target_bit_index_right = std::countr_zero(target_mask);
    const std::uint64_t target_bit_index_left = 64 - std::countl_zero(target_mask | 1 << 63) - 1;
    const std::uint64_t target_bit_right = 1ULL << target_bit_index_right;
    const std::uint64_t target_bit_left = 1ULL << target_bit_index_left;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_bit_right;
            std::uint64_t basis_2 = basis_0 | target_bit_left;
            std::uint64_t basis_3 = basis_0 | target_mask;
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

void multi_control_multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                                  std::uint64_t control_mask,
                                                  const Matrix& matrix,
                                                  StateVector& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    const std::uint64_t matrix_dim = 1ULL << target_qubit_index_count;

    Kokkos::View<Complex*> update(Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask =
        ~target_mask & ((1ULL << state.n_qubits()) - 1);  // target qubit 以外の mask
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(state.dim() >> std::popcount(target_mask | control_mask),
                             Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) | control_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
                uint32_t dst_index = internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t c, Complex& inner_sum) {
                        uint32_t src_index =
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

void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix& matrix,
                       StateVector& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    if (target_qubit_index_count == 1) {
        multi_control_single_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
    } else if (target_qubit_index_count == 2) {
        multi_control_double_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
    } else {
        multi_control_multi_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
    }
}
}  // namespace internal
}  // namespace scaluq
