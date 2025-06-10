#include "../prec_space.hpp"
#include "../util/math.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec>
Matrix2x2<Prec> get_IBMQ_matrix(Float<Prec> _theta, Float<Prec> _phi, Float<Prec> _lambda) {
    Complex<Prec> exp_val1 = internal::exp(Complex<Prec>(0, _phi));
    Complex<Prec> exp_val2 = internal::exp(Complex<Prec>(0, _lambda));
    Complex<Prec> cos_val = internal::cos(_theta / Float<Prec>{2});
    Complex<Prec> sin_val = internal::sin(_theta / Float<Prec>{2});
    return {cos_val, -exp_val2 * sin_val, exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val};
}

template <>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
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

template <>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
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

template <>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVector<Prec, Space>& state) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        "two_target_dense_matrix_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
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

template <>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVectorBatched<Prec, Space>& states) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        "two_target_dense_matrix_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
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

template <>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "one_target_diagonal_matrix_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            state._raw[basis] *= diag[0];
            state._raw[basis | target_mask] *= diag[1];
        });
    Kokkos::fence();
}

template <>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "one_target_diagonal_matrix_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            states._raw(batch_id, basis) *= diag[0];
            states._raw(batch_id, basis | target_mask) *= diag[1];
        });
    Kokkos::fence();
}

template <>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       StateVector<Prec, Space>& state) {
    Complex<Prec> coef = internal::polar<Prec>(Float<Prec>{1}, angle);
    Kokkos::parallel_for(
        "global_phase_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(0, state.dim() >> std::popcount(control_mask)),
        KOKKOS_LAMBDA(std::uint64_t i) {
            state._raw[insert_zero_at_mask_positions(i, control_mask) | control_value_mask] *= coef;
        });
    Kokkos::fence();
}

template <>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       StateVectorBatched<Prec, Space>& states) {
    Complex<Prec> coef = internal::polar<Prec>(Float<Prec>{1}, angle);
    Kokkos::parallel_for(
        "global_phase_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            states._raw(batch_id,
                        insert_zero_at_mask_positions(i, control_mask) | control_value_mask) *=
                coef;
        });
    Kokkos::fence();
}

template <>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "x_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            Kokkos::kokkos_swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

template <>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "x_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            Kokkos::kokkos_swap(states._raw(batch_id, i), states._raw(batch_id, i | target_mask));
        });
    Kokkos::fence();
}

template <>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "y_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            state._raw[i] *= Complex<Prec>(0, 1);
            state._raw[i | target_mask] *= Complex<Prec>(0, -1);
            Kokkos::kokkos_swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

template <>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "y_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            states._raw(batch_id, i) *= Complex<Prec>(0, 1);
            states._raw(batch_id, i | target_mask) *= Complex<Prec>(0, -1);
            Kokkos::kokkos_swap(states._raw(batch_id, i), states._raw(batch_id, i | target_mask));
        });
    Kokkos::fence();
}

template <>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "z_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            state._raw[i | target_mask] *= Complex<Prec>(-1, 0);
        });
    Kokkos::fence();
}

template <>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "z_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            states._raw(batch_id, i | target_mask) *= Complex<Prec>(-1, 0);
        });
    Kokkos::fence();
}

template <>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           StateVector<Prec, Space>& state) {
    Kokkos::parallel_for(
        "one_target_phase_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            state._raw[i | target_mask] *= phase;
        });
    Kokkos::fence();
}

template <>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           StateVectorBatched<Prec, Space>& states) {
    Kokkos::parallel_for(
        "one_target_phase_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            states._raw(batch_id, i | target_mask) *= phase;
        });
    Kokkos::fence();
}

template <>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVector<Prec, Space>& state) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, Complex<Prec>(0, -sinval), Complex<Prec>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, state);
}

template <>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec, Space>& states) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, Complex<Prec>(0, -sinval), Complex<Prec>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, states);
}

template <>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec, Space>& states) {
    auto team_policy =
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        "rx_gate",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Float<Prec> angle = params[batch_id] * pcoef;
            const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
            const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
            Matrix2x2<Prec> matrix = {
                cosval, Complex<Prec>(0, -sinval), Complex<Prec>(0, -sinval), cosval};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_value_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    Complex<Prec> val0 = states._raw(batch_id, basis_0);
                    Complex<Prec> val1 = states._raw(batch_id, basis_1);
                    Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
                    Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
                    states._raw(batch_id, basis_0) = res0;
                    states._raw(batch_id, basis_1) = res1;
                });
        });
    Kokkos::fence();
}

template <>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVector<Prec, Space>& state) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, state);
}

template <>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec, Space>& states) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, states);
}

template <>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec, Space>& states) {
    auto team_policy =
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        "ry_gate",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Float<Prec> angle = params[batch_id] * pcoef;
            const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
            const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
            Matrix2x2<Prec> matrix = {cosval, -sinval, sinval, cosval};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_value_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    Complex<Prec> val0 = states._raw(batch_id, basis_0);
                    Complex<Prec> val1 = states._raw(batch_id, basis_1);
                    Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
                    Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
                    states._raw(batch_id, basis_0) = res0;
                    states._raw(batch_id, basis_1) = res1;
                });
        });
    Kokkos::fence();
}

template <>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVector<Prec, Space>& state) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(cosval, -sinval), Complex<Prec>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, control_value_mask, diag, state);
}

template <>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec, Space>& states) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(cosval, -sinval), Complex<Prec>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, control_value_mask, diag, states);
}

template <>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec, Space>& states) {
    auto team_policy =
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        "rz_gate",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Float<Prec> angle = params[batch_id] * pcoef;
            const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
            const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
            DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(cosval, -sinval),
                                            Complex<Prec>(cosval, sinval)};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_value_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    states._raw(batch_id, basis_0) *= diag[0];
                    states._raw(batch_id, basis_1) *= diag[1];
                });
        });
    Kokkos::fence();
}

template <>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> lambda,
             StateVector<Prec, Space>& state) {
    Complex<Prec> exp_val = internal::exp(Complex<Prec>(0, lambda));
    Kokkos::parallel_for(
        "u1_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> (std::popcount(target_mask | control_mask))),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                internal::insert_zero_at_mask_positions(it, target_mask | control_mask) |
                control_value_mask;
            state._raw[i | target_mask] *= exp_val;
        });
    Kokkos::fence();
}

template <>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> lambda,
             StateVectorBatched<Prec, Space>& states) {
    Complex<Prec> exp_val = internal::exp(Complex<Prec>(0, lambda));
    Kokkos::parallel_for(
        "u1_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                internal::insert_zero_at_mask_positions(it, target_mask | control_mask) |
                control_value_mask;
            states._raw(batch_id, i | target_mask) *= exp_val;
        });
    Kokkos::fence();
}

template <>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        control_value_mask,
        get_IBMQ_matrix<Prec>(static_cast<Float<Prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        state);
}

template <>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        control_value_mask,
        get_IBMQ_matrix<Prec>(static_cast<Float<Prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        states);
}

template <>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 get_IBMQ_matrix<Prec>(theta, phi, lambda),
                                 state);
}

template <>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 get_IBMQ_matrix<Prec>(theta, phi, lambda),
                                 states);
}

template <>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               StateVector<Prec, Space>& state) {
    // '- target' is used for bit manipulation on unsigned type, not for its numerical meaning.
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        "swap_gate",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            Kokkos::kokkos_swap(state._raw[basis | lower_target_mask],
                                state._raw[basis | upper_target_mask]);
        });
    Kokkos::fence();
}

template <>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               StateVectorBatched<Prec, Space>& states) {
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        "swap_gate",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            Kokkos::kokkos_swap(states._raw(batch_id, basis | lower_target_mask),
                                states._raw(batch_id, basis | upper_target_mask));
        });
    Kokkos::fence();
}
}  // namespace scaluq::internal
