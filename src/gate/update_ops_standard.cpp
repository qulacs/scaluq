#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <std::floating_point Fp>
Matrix2x2<Fp> get_IBMQ_matrix(Fp _theta, Fp _phi, Fp _lambda) {
    Complex<Fp> exp_val1 = Kokkos::exp(Complex<Fp>(0, _phi));
    Complex<Fp> exp_val2 = Kokkos::exp(Complex<Fp>(0, _lambda));
    Complex<Fp> cos_val = Kokkos::cos(_theta / 2.);
    Complex<Fp> sin_val = Kokkos::sin(_theta / 2.);
    return {cos_val, -exp_val2 * sin_val, exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val};
}
#define FUNC_MACRO(Fp) template Matrix2x2<Fp> get_IBMQ_matrix(Fp, Fp, Fp);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

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
#define FUNC_MACRO(Fp)                          \
    template void one_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix2x2<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Fp>& matrix,
                                  StateVectorBatched<Fp>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<Fp> val0 = states._raw(batch_id, basis_0);
            Complex<Fp> val1 = states._raw(batch_id, basis_1);
            Complex<Fp> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex<Fp> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            states._raw(batch_id, basis_0) = res0;
            states._raw(batch_id, basis_1) = res1;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)                          \
    template void one_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix2x2<Fp>&, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

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
#define FUNC_MACRO(Fp)                          \
    template void two_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix4x4<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Fp>& matrix,
                                  StateVectorBatched<Fp>& states) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
            Complex<Fp> val0 = states._raw(batch_id, basis_0);
            Complex<Fp> val1 = states._raw(batch_id, basis_1);
            Complex<Fp> val2 = states._raw(batch_id, basis_2);
            Complex<Fp> val3 = states._raw(batch_id, basis_3);
            Complex<Fp> res0 = matrix[0][0] * val0 + matrix[0][1] * val1 + matrix[0][2] * val2 +
                               matrix[0][3] * val3;
            Complex<Fp> res1 = matrix[1][0] * val0 + matrix[1][1] * val1 + matrix[1][2] * val2 +
                               matrix[1][3] * val3;
            Complex<Fp> res2 = matrix[2][0] * val0 + matrix[2][1] * val1 + matrix[2][2] * val2 +
                               matrix[2][3] * val3;
            Complex<Fp> res3 = matrix[3][0] * val0 + matrix[3][1] * val1 + matrix[3][2] * val2 +
                               matrix[3][3] * val3;
            states._raw(batch_id, basis_0) = res0;
            states._raw(batch_id, basis_1) = res1;
            states._raw(batch_id, basis_2) = res2;
            states._raw(batch_id, basis_3) = res3;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)                          \
    template void two_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix4x4<Fp>&, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
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
#define FUNC_MACRO(Fp)                             \
    template void one_target_diagonal_matrix_gate( \
        std::uint64_t, std::uint64_t, const DiagonalMatrix2x2<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVectorBatched<Fp>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            states._raw(batch_id, basis) *= diag[0];
            states._raw(batch_id, basis | target_mask) *= diag[1];
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)                             \
    template void one_target_diagonal_matrix_gate( \
        std::uint64_t, std::uint64_t, const DiagonalMatrix2x2<Fp>&, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void global_phase_gate(std::uint64_t,
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
#define FUNC_MACRO(Fp) \
    template void global_phase_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Fp angle,
                       StateVectorBatched<Fp>& states) {
    Complex<Fp> coef = Kokkos::polar<Fp>(1., angle);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            states._raw(batch_id, insert_zero_at_mask_positions(i, control_mask) | control_mask) *=
                coef;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) \
    template void global_phase_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) template void x_gate(std::uint64_t, std::uint64_t, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched<Fp>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(states._raw(batch_id, i),
                                       states._raw(batch_id, i | target_mask));
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) template void x_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
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
#define FUNC_MACRO(Fp) template void y_gate(std::uint64_t, std::uint64_t, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched<Fp>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            states._raw(batch_id, i) *= Complex<Fp>(0, 1);
            states._raw(batch_id, i | target_mask) *= Complex<Fp>(0, -1);
            Kokkos::Experimental::swap(states._raw(batch_id, i),
                                       states._raw(batch_id, i | target_mask));
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) template void y_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= Complex<Fp>(-1, 0);
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) template void z_gate(std::uint64_t, std::uint64_t, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched<Fp>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            states._raw(batch_id, i | target_mask) *= Complex<Fp>(-1, 0);
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) template void z_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void one_target_phase_gate(std::uint64_t target_mask,
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
#define FUNC_MACRO(Fp)                   \
    template void one_target_phase_gate( \
        std::uint64_t, std::uint64_t, Complex<Fp>, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVectorBatched<Fp>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            states._raw(batch_id, i | target_mask) *= phase;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)                   \
    template void one_target_phase_gate( \
        std::uint64_t, std::uint64_t, Complex<Fp>, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, Complex<Fp>(0, -sinval), Complex<Fp>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}
#define FUNC_MACRO(Fp) template void rx_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp>& states) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, Complex<Fp>(0, -sinval), Complex<Fp>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, states);
}
#define FUNC_MACRO(Fp) \
    template void rx_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp>& states) {
    auto team_policy = Kokkos::TeamPolicy<>(states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Fp angle = params[batch_id] * pcoef;
            const Fp cosval = std::cos(angle / 2.);
            const Fp sinval = std::sin(angle / 2.);
            Matrix2x2<Fp> matrix = {
                cosval, Complex<Fp>(0, -sinval), Complex<Fp>(0, -sinval), cosval};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    Complex<Fp> val0 = states._raw(batch_id, basis_0);
                    Complex<Fp> val1 = states._raw(batch_id, basis_1);
                    Complex<Fp> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
                    Complex<Fp> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
                    states._raw(batch_id, basis_0) = res0;
                    states._raw(batch_id, basis_1) = res1;
                });
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)     \
    template void rx_gate( \
        std::uint64_t, std::uint64_t, Fp, std::vector<Fp>, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}
#define FUNC_MACRO(Fp) template void ry_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp>& states) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, states);
}
#define FUNC_MACRO(Fp) \
    template void ry_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp>& states) {
    auto team_policy = Kokkos::TeamPolicy<>(states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Fp angle = params[batch_id] * pcoef;
            const Fp cosval = std::cos(angle / 2.);
            const Fp sinval = std::sin(angle / 2.);
            Matrix2x2<Fp> matrix = {cosval, -sinval, sinval, cosval};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    Complex<Fp> val0 = states._raw(batch_id, basis_0);
                    Complex<Fp> val1 = states._raw(batch_id, basis_1);
                    Complex<Fp> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
                    Complex<Fp> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
                    states._raw(batch_id, basis_0) = res0;
                    states._raw(batch_id, basis_1) = res1;
                });
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)     \
    template void ry_gate( \
        std::uint64_t, std::uint64_t, Fp, std::vector<Fp>, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    DiagonalMatrix2x2<Fp> diag = {Complex<Fp>(cosval, -sinval), Complex<Fp>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, state);
}
#define FUNC_MACRO(Fp) template void rz_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp>& states) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    DiagonalMatrix2x2<Fp> diag = {Complex<Fp>(cosval, -sinval), Complex<Fp>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, states);
}
#define FUNC_MACRO(Fp) \
    template void rz_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp>& states) {
    auto team_policy = Kokkos::TeamPolicy<>(states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Fp angle = params[batch_id] * pcoef;
            const Fp cosval = std::cos(angle / 2.);
            const Fp sinval = std::sin(angle / 2.);
            DiagonalMatrix2x2<Fp> diag = {Complex<Fp>(cosval, -sinval),
                                          Complex<Fp>(cosval, sinval)};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    states._raw(batch_id, basis_0) *= diag[0];
                    states._raw(batch_id, basis_1) *= diag[1];
                });
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)     \
    template void rz_gate( \
        std::uint64_t, std::uint64_t, Fp, std::vector<Fp>, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void u1_gate(std::uint64_t target_mask,
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
#define FUNC_MACRO(Fp) template void u1_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVectorBatched<Fp>& states) {
    Complex<Fp> exp_val = Kokkos::exp(Complex<Fp>(0, lambda));
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                internal::insert_zero_at_mask_positions(it, target_mask | control_mask) |
                control_mask;
            states._raw(batch_id, i | target_mask) *= exp_val;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) \
    template void u1_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 get_IBMQ_matrix((Fp)Kokkos::numbers::pi / 2, phi, lambda),
                                 state);
}
#define FUNC_MACRO(Fp) \
    template void u2_gate(std::uint64_t, std::uint64_t, Fp, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 get_IBMQ_matrix((Fp)Kokkos::numbers::pi / 2, phi, lambda),
                                 states);
}
#define FUNC_MACRO(Fp) \
    template void u2_gate(std::uint64_t, std::uint64_t, Fp, Fp, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVector<Fp>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(theta, phi, lambda), state);
}
#define FUNC_MACRO(Fp) \
    template void u3_gate(std::uint64_t, std::uint64_t, Fp, Fp, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(theta, phi, lambda), states);
}
#define FUNC_MACRO(Fp) \
    template void u3_gate(std::uint64_t, std::uint64_t, Fp, Fp, Fp, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
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
#define FUNC_MACRO(Fp) template void swap_gate(std::uint64_t, std::uint64_t, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVectorBatched<Fp>& states) {
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            Kokkos::Experimental::swap(states._raw(batch_id, basis | lower_target_mask),
                                       states._raw(batch_id, basis | upper_target_mask));
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp) \
    template void swap_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO
}  // namespace scaluq::internal
