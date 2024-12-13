#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <FloatingPoint Fp>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Fp>& matrix,
                                   StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis = insert_zero_at_mask_positions(it, control_mask) | control_mask;
            state._raw[basis] *= matrix(0, 0);
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) \
    template void none_target_dense_matrix_gate(std::uint64_t, const Matrix<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <FloatingPoint Fp>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<Fp> val0 = state._raw[basis_0];
            Complex<Fp> val1 = state._raw[basis_1];
            Complex<Fp> res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1;
            Complex<Fp> res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)                             \
    template void single_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <FloatingPoint Fp>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp>& state) {
    std::uint64_t target_bit_right = -target_mask & target_mask;
    std::uint64_t target_bit_left = target_mask ^ target_bit_right;

    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_bit_right;
            std::uint64_t basis_2 = basis_0 | target_bit_left;
            std::uint64_t basis_3 = basis_0 | target_mask;
            Complex<Fp> val0 = state._raw[basis_0];
            Complex<Fp> val1 = state._raw[basis_1];
            Complex<Fp> val2 = state._raw[basis_2];
            Complex<Fp> val3 = state._raw[basis_3];
            Complex<Fp> res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1 + matrix(0, 2) * val2 +
                               matrix(0, 3) * val3;
            Complex<Fp> res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1 + matrix(1, 2) * val2 +
                               matrix(1, 3) * val3;
            Complex<Fp> res2 = matrix(2, 0) * val0 + matrix(2, 1) * val1 + matrix(2, 2) * val2 +
                               matrix(2, 3) * val3;
            Complex<Fp> res3 = matrix(3, 0) * val0 + matrix(3, 1) * val1 + matrix(3, 2) * val2 +
                               matrix(3, 3) * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)                             \
    template void double_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <FloatingPoint Fp>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Fp>& matrix,
                                    StateVector<Fp>& state) {
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);

    Kokkos::View<Complex<Fp>*> update(Kokkos::ViewAllocateWithoutInitializing("update"),
                                      state.dim());
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(state.dim() >> std::popcount(target_mask | control_mask),
                             Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) | control_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
                std::uint64_t dst_index =
                    internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex<Fp> sum = Fp{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t c, Complex<Fp>& inner_sum) {
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
#define FUNC_MACRO(Fp)                            \
    template void multi_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <FloatingPoint Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVector<Fp>& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    if (target_qubit_index_count == 0) {
        none_target_dense_matrix_gate<Fp>(control_mask, matrix, state);
    } else if (target_qubit_index_count == 1) {
        single_target_dense_matrix_gate<Fp>(target_mask, control_mask, matrix, state);
    } else if (target_qubit_index_count == 2) {
        double_target_dense_matrix_gate<Fp>(target_mask, control_mask, matrix, state);
    } else {
        multi_target_dense_matrix_gate<Fp>(target_mask, control_mask, matrix, state);
    }
}
#define FUNC_MACRO(Fp)               \
    template void dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO
}  // namespace scaluq::internal
