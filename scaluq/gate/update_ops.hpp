
#pragma once

#include "../operator/pauli_operator.hpp"
#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp>
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

template <std::floating_point Fp>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Fp>& matrix,
                                  StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<Fp> val0 = state._raw[basis_0];
            Complex<Fp> val1 = state._raw[basis_1];
            Complex<Fp> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex<Fp> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Fp>& matrix,
                                  StateVector<Fp>& state) {
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
            Complex<Fp> val0 = state._raw[basis_0];
            Complex<Fp> val1 = state._raw[basis_1];
            Complex<Fp> val2 = state._raw[basis_2];
            Complex<Fp> val3 = state._raw[basis_3];
            Complex<Fp> res0 = matrix[0][0] * val0 + matrix[0][1] * val1 + matrix[0][2] * val2 +
                               matrix[0][3] * val3;
            Complex<Fp> res1 = matrix[1][0] * val0 + matrix[1][1] * val1 + matrix[1][2] * val2 +
                               matrix[1][3] * val3;
            Complex<Fp> res2 = matrix[2][0] * val0 + matrix[2][1] * val1 + matrix[2][2] * val2 +
                               matrix[2][3] * val3;
            Complex<Fp> res3 = matrix[3][0] * val0 + matrix[3][1] * val1 + matrix[3][2] * val2 +
                               matrix[3][3] * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
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

template <std::floating_point Fp>
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

template <std::floating_point Fp>
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
                Complex<Fp> sum = 0;
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

template <std::floating_point Fp>
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

template <std::floating_point Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& mat,
                        StateVector<Fp>& state) {
    auto values = mat._values;

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
    Kokkos::View<Complex<Fp>*, Kokkos::MemoryTraits<Kokkos::Atomic>> update_atomic(update);
    Kokkos::parallel_for(
        "COO_Update",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {static_cast<std::int64_t>(state.dim() >> std::popcount(target_mask | control_mask)),
             static_cast<std::int64_t>(values.size())}),
        KOKKOS_LAMBDA(std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_mask;
            auto [v, r, c] = values(inner);
            uint32_t src_index = internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            uint32_t dst_index = internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            update_atomic(dst_index) += v * state._raw(src_index);
        });
    Kokkos::fence();
    state._raw = update;
}

template <std::floating_point Fp>
inline Matrix2x2<Fp> get_IBMQ_matrix(Fp _theta, Fp _phi, Fp _lambda) {
    Complex<Fp> exp_val1 = Kokkos::exp(Complex<Fp>(0, _phi));
    Complex<Fp> exp_val2 = Kokkos::exp(Complex<Fp>(0, _lambda));
    Complex<Fp> cos_val = Kokkos::cos(_theta / 2.);
    Complex<Fp> sin_val = Kokkos::sin(_theta / 2.);
    return {cos_val, -exp_val2 * sin_val, exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val};
}

template <std::floating_point Fp>
inline void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                            std::uint64_t control_mask,
                                            const DiagonalMatrix2x2<Fp>& diag,
                                            StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            state._raw[basis] *= diag[0];
            state._raw[basis | target_mask] *= diag[1];
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
inline void i_gate(std::uint64_t, std::uint64_t, StateVector<Fp>&) {}

template <std::floating_point Fp>
inline void global_phase_gate(std::uint64_t,
                              std::uint64_t control_mask,
                              Fp angle,
                              StateVector<Fp>& state) {
    Complex<Fp> coef = Kokkos::polar<Fp>(1., angle);
    Kokkos::parallel_for(
        state.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t i) {
            state._raw[insert_zero_at_mask_positions(i, control_mask) | control_mask] *= coef;
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
inline void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
inline void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i] *= Complex<Fp>(0, 1);
            state._raw[i | target_mask] *= Complex<Fp>(0, -1);
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
inline void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= Complex<Fp>(-1, 0);
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
inline void h_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Fp>(), state);
}

template <std::floating_point Fp>
inline void one_target_phase_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  Complex<Fp> phase,
                                  StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= phase;
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
inline void s_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, 1), state);
}

template <std::floating_point Fp>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, -1), state);
}

template <std::floating_point Fp>
inline void t_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}

template <std::floating_point Fp>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}

template <std::floating_point Fp>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Fp>(), state);
}

template <std::floating_point Fp>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Fp>(), state);
}

template <std::floating_point Fp>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Fp>(), state);
}

template <std::floating_point Fp>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Fp>(), state);
}

template <std::floating_point Fp>
inline void p0_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Fp>(), state);
}

template <std::floating_point Fp>
inline void p1_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Fp>(), state);
}

template <std::floating_point Fp>
inline void rx_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    Fp angle,
                    StateVector<Fp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, Complex<Fp>(0, -sinval), Complex<Fp>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

template <std::floating_point Fp>
inline void ry_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    Fp angle,
                    StateVector<Fp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}

template <std::floating_point Fp>
inline void rz_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    Fp angle,
                    StateVector<Fp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    DiagonalMatrix2x2<Fp> diag = {Complex<Fp>(cosval, -sinval), Complex<Fp>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, state);
}

template <std::floating_point Fp>
inline void u1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    Fp lambda,
                    StateVector<Fp>& state) {
    Complex<Fp> exp_val = Kokkos::exp(Complex<Fp>(0, lambda));
    Kokkos::parallel_for(
        state.dim() >> (std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                internal::insert_zero_at_mask_positions(it, target_mask | control_mask) |
                control_mask;
            state._raw[i | target_mask] *= exp_val;
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
inline void u2_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    Fp phi,
                    Fp lambda,
                    StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 get_IBMQ_matrix((Fp)Kokkos::numbers::pi / 2, phi, lambda),
                                 state);
}

template <std::floating_point Fp>
inline void u3_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    Fp theta,
                    Fp phi,
                    Fp lambda,
                    StateVector<Fp>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(theta, phi, lambda), state);
}

template <std::floating_point Fp>
inline void swap_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp>& state) {
    // '- target' is used for bit manipulation on unsigned type, not for its numerical meaning.
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[basis | lower_target_mask],
                                       state._raw[basis | upper_target_mask]);
        });
    Kokkos::fence();
}

template <std::floating_point Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& matrix,
                        StateVector<Fp>& state);

template <std::floating_point Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVector<Fp>& state);
}  // namespace internal
}  // namespace scaluq
