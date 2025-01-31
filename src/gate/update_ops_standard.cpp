#include "../util/math.hpp"
#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
template <Precision Prec>
Matrix2x2<Prec> get_IBMQ_matrix(Float<Prec> _theta, Float<Prec> _phi, Float<Prec> _lambda) {
    Complex<Prec> exp_val1 = internal::exp(Complex<Prec>(0, _phi));
    Complex<Prec> exp_val2 = internal::exp(Complex<Prec>(0, _lambda));
    Complex<Prec> cos_val = internal::cos(_theta / Float<Prec>{2});
    Complex<Prec> sin_val = internal::sin(_theta / Float<Prec>{2});
=======
FLOAT(Fp)
Matrix2x2<Fp> get_IBMQ_matrix(Fp _theta, Fp _phi, Fp _lambda) {
    Complex<Fp> exp_val1 = Kokkos::exp(Complex<Fp>(0, _phi));
    Complex<Fp> exp_val2 = Kokkos::exp(Complex<Fp>(0, _lambda));
    Complex<Fp> cos_val = Kokkos::cos(_theta / 2.);
    Complex<Fp> sin_val = Kokkos::sin(_theta / 2.);
>>>>>>> set-space
    return {cos_val, -exp_val2 * sin_val, exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val};
}
#define FUNC_MACRO(Prec) \
    template Matrix2x2<Prec> get_IBMQ_matrix(Float<Prec>, Float<Prec>, Float<Prec>);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

<<<<<<< HEAD
template <Precision Prec>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVector<Prec>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
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
#define FUNC_MACRO(Prec)                        \
    template void one_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix2x2<Prec>&, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVectorBatched<Prec>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
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
#define FUNC_MACRO(Prec)                        \
    template void one_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix2x2<Prec>&, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVector<Prec>& state) {
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
#define FUNC_MACRO(Prec)                        \
    template void two_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix4x4<Prec>&, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVectorBatched<Prec>& states) {
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
#define FUNC_MACRO(Prec)                        \
    template void two_target_dense_matrix_gate( \
        std::uint64_t, std::uint64_t, const Matrix4x4<Prec>&, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVector<Prec>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            state._raw[basis] *= diag[0];
            state._raw[basis | target_mask] *= diag[1];
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Prec)                           \
    template void one_target_diagonal_matrix_gate( \
        std::uint64_t, std::uint64_t, const DiagonalMatrix2x2<Prec>&, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVectorBatched<Prec>& states) {
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
#define FUNC_MACRO(Prec)                           \
    template void one_target_diagonal_matrix_gate( \
        std::uint64_t, std::uint64_t, const DiagonalMatrix2x2<Prec>&, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Float<Prec> angle,
                       StateVector<Prec>& state) {
    Complex<Prec> coef = internal::polar<Prec>(Float<Prec>{1}, angle);
=======
FLOAT_AND_SPACE(Fp, Sp)
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Fp angle,
                       StateVector<Fp, Sp>& state) {
    Complex<Fp> coef = Kokkos::polar<Fp>(1., angle);
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> std::popcount(control_mask)),
        KOKKOS_LAMBDA(std::uint64_t i) {
            state._raw[insert_zero_at_mask_positions(i, control_mask) | control_mask] *= coef;
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template void global_phase_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Float<Prec> angle,
                       StateVectorBatched<Prec>& states) {
    Complex<Prec> coef = internal::polar<Prec>(Float<Prec>{1}, angle);
=======
#define FUNC_MACRO(Fp, Sp) \
    template void global_phase_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Fp angle,
                       StateVectorBatched<Fp, Sp>& states) {
    Complex<Fp> coef = Kokkos::polar<Fp>(1., angle);
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            states._raw(batch_id, insert_zero_at_mask_positions(i, control_mask) | control_mask) *=
                coef;
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec)             \
    template void global_phase_gate( \
        std::uint64_t, std::uint64_t, Float<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void global_phase_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) template void x_gate(std::uint64_t, std::uint64_t, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Prec>& states) {
=======
#define FUNC_MACRO(Fp, Sp) template void x_gate(std::uint64_t, std::uint64_t, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
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
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template void x_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void x_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i] *= Complex<Prec>(0, 1);
            state._raw[i | target_mask] *= Complex<Prec>(0, -1);
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) template void y_gate(std::uint64_t, std::uint64_t, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Prec>& states) {
=======
#define FUNC_MACRO(Fp, Sp) template void y_gate(std::uint64_t, std::uint64_t, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            states._raw(batch_id, i) *= Complex<Prec>(0, 1);
            states._raw(batch_id, i | target_mask) *= Complex<Prec>(0, -1);
            Kokkos::Experimental::swap(states._raw(batch_id, i),
                                       states._raw(batch_id, i | target_mask));
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template void y_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void y_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= Complex<Prec>(-1, 0);
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) template void z_gate(std::uint64_t, std::uint64_t, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Prec>& states) {
=======
#define FUNC_MACRO(Fp, Sp) template void z_gate(std::uint64_t, std::uint64_t, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            states._raw(batch_id, i | target_mask) *= Complex<Prec>(-1, 0);
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template void z_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Prec> phase,
                           StateVector<Prec>& state) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void z_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVector<Fp, Sp>& state) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            state._raw[i | target_mask] *= phase;
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec)                 \
    template void one_target_phase_gate( \
        std::uint64_t, std::uint64_t, Complex<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Prec> phase,
                           StateVectorBatched<Prec>& states) {
=======
#define FUNC_MACRO(Fp, Sp)               \
    template void one_target_phase_gate( \
        std::uint64_t, std::uint64_t, Complex<Fp>, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            states._raw(batch_id, i | target_mask) *= phase;
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec)                 \
    template void one_target_phase_gate( \
        std::uint64_t, std::uint64_t, Complex<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVector<Prec>& state) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, Complex<Prec>(0, -sinval), Complex<Prec>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}
#define FUNC_MACRO(Prec) \
    template void rx_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec>& states) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, Complex<Prec>(0, -sinval), Complex<Prec>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, states);
}
#define FUNC_MACRO(Prec) \
    template void rx_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec>& states) {
    auto team_policy = Kokkos::TeamPolicy<>(states.batch_size(), Kokkos::AUTO);
=======
#define FUNC_MACRO(Fp, Sp)               \
    template void one_target_phase_gate( \
        std::uint64_t, std::uint64_t, Complex<Fp>, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp, Sp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, Complex<Fp>(0, -sinval), Complex<Fp>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}
#define FUNC_MACRO(Fp, Sp) \
    template void rx_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp, Sp>& states) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, Complex<Fp>(0, -sinval), Complex<Fp>(0, -sinval), cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, states);
}
#define FUNC_MACRO(Fp, Sp) \
    template void rx_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp, Sp>& states) {
    Kokkos::TeamPolicy<Sp> team_policy(Sp(), states.batch_size(), Kokkos::AUTO);
>>>>>>> set-space
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<Sp>::member_type& team_member) {
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
                        control_mask;
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
<<<<<<< HEAD
#define FUNC_MACRO(Prec)                            \
    template void rx_gate(std::uint64_t,            \
                          std::uint64_t,            \
                          Float<Prec>,              \
                          std::vector<Float<Prec>>, \
                          StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVector<Prec>& state) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}
#define FUNC_MACRO(Prec) \
    template void ry_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec>& states) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    Matrix2x2<Prec> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, states);
}
#define FUNC_MACRO(Prec) \
    template void ry_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec>& states) {
    auto team_policy = Kokkos::TeamPolicy<>(states.batch_size(), Kokkos::AUTO);
=======
#define FUNC_MACRO(Fp, Sp) \
    template void rx_gate( \
        std::uint64_t, std::uint64_t, Fp, std::vector<Fp>, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp, Sp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, state);
}
#define FUNC_MACRO(Fp, Sp) \
    template void ry_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp, Sp>& states) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    Matrix2x2<Fp> matrix = {cosval, -sinval, sinval, cosval};
    one_target_dense_matrix_gate(target_mask, control_mask, matrix, states);
}
#define FUNC_MACRO(Fp, Sp) \
    template void ry_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp, Sp>& states) {
    Kokkos::TeamPolicy<Sp> team_policy(Sp(), states.batch_size(), Kokkos::AUTO);
>>>>>>> set-space
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<Sp>::member_type& team_member) {
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
                        control_mask;
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
<<<<<<< HEAD
#define FUNC_MACRO(Prec)                            \
    template void ry_gate(std::uint64_t,            \
                          std::uint64_t,            \
                          Float<Prec>,              \
                          std::vector<Float<Prec>>, \
                          StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVector<Prec>& state) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(cosval, -sinval), Complex<Prec>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, state);
}
#define FUNC_MACRO(Prec) \
    template void rz_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec>& states) {
    const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
    const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
    DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(cosval, -sinval), Complex<Prec>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, states);
}
#define FUNC_MACRO(Prec) \
    template void rz_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec>& states) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void ry_gate( \
        std::uint64_t, std::uint64_t, Fp, std::vector<Fp>, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp, Sp>& state) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    DiagonalMatrix2x2<Fp> diag = {Complex<Fp>(cosval, -sinval), Complex<Fp>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, state);
}
#define FUNC_MACRO(Fp, Sp) \
    template void rz_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp, Sp>& states) {
    const Fp cosval = std::cos(angle / 2.);
    const Fp sinval = std::sin(angle / 2.);
    DiagonalMatrix2x2<Fp> diag = {Complex<Fp>(cosval, -sinval), Complex<Fp>(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, diag, states);
}
#define FUNC_MACRO(Fp, Sp) \
    template void rz_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    auto team_policy = Kokkos::TeamPolicy<>(states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
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
                        control_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    states._raw(batch_id, basis_0) *= diag[0];
                    states._raw(batch_id, basis_1) *= diag[1];
                });
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec)                            \
    template void rz_gate(std::uint64_t,            \
                          std::uint64_t,            \
                          Float<Prec>,              \
                          std::vector<Float<Prec>>, \
                          StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> lambda,
             StateVector<Prec>& state) {
    Complex<Prec> exp_val = internal::exp(Complex<Prec>(0, lambda));
=======
#define FUNC_MACRO(Fp, Sp) \
    template void rz_gate( \
        std::uint64_t, std::uint64_t, Fp, std::vector<Fp>, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVector<Fp, Sp>& state) {
    Complex<Fp> exp_val = Kokkos::exp(Complex<Fp>(0, lambda));
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> (std::popcount(target_mask | control_mask))),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                internal::insert_zero_at_mask_positions(it, target_mask | control_mask) |
                control_mask;
            state._raw[i | target_mask] *= exp_val;
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template void u1_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> lambda,
             StateVectorBatched<Prec>& states) {
    Complex<Prec> exp_val = internal::exp(Complex<Prec>(0, lambda));
=======
#define FUNC_MACRO(Fp, Sp) \
    template void u1_gate(std::uint64_t, std::uint64_t, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVectorBatched<Fp, Sp>& states) {
    Complex<Fp> exp_val = Kokkos::exp(Complex<Fp>(0, lambda));
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
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
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template void u1_gate(std::uint64_t, std::uint64_t, Float<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec>& state) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        get_IBMQ_matrix<Prec>(static_cast<Float<Prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        state);
}
#define FUNC_MACRO(Prec)   \
    template void u2_gate( \
        std::uint64_t, std::uint64_t, Float<Prec>, Float<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        get_IBMQ_matrix<Prec>(static_cast<Float<Prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        states);
}
#define FUNC_MACRO(Prec)   \
    template void u2_gate( \
        std::uint64_t, std::uint64_t, Float<Prec>, Float<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec>& state) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void u1_gate(std::uint64_t, std::uint64_t, Fp, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 get_IBMQ_matrix((Fp)Kokkos::numbers::pi / 2, phi, lambda),
                                 state);
}
#define FUNC_MACRO(Fp, Sp) \
    template void u2_gate(std::uint64_t, std::uint64_t, Fp, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 get_IBMQ_matrix((Fp)Kokkos::numbers::pi / 2, phi, lambda),
                                 states);
}
#define FUNC_MACRO(Fp, Sp) \
    template void u2_gate(std::uint64_t, std::uint64_t, Fp, Fp, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVector<Fp, Sp>& state) {
>>>>>>> set-space
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix<Prec>(theta, phi, lambda), state);
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec)   \
    template void u3_gate( \
        std::uint64_t, std::uint64_t, Float<Prec>, Float<Prec>, Float<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec>& states) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void u3_gate(std::uint64_t, std::uint64_t, Fp, Fp, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    one_target_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix<Prec>(theta, phi, lambda), states);
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec)                 \
    template void u3_gate(std::uint64_t, \
                          std::uint64_t, \
                          Float<Prec>,   \
                          Float<Prec>,   \
                          Float<Prec>,   \
                          StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void u3_gate(std::uint64_t, std::uint64_t, Fp, Fp, Fp, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state) {
>>>>>>> set-space
    // '- target' is used for bit manipulation on unsigned type, not for its numerical meaning.
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[basis | lower_target_mask],
                                       state._raw[basis | upper_target_mask]);
        });
    Kokkos::fence();
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) template void swap_gate(std::uint64_t, std::uint64_t, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVectorBatched<Prec>& states) {
=======
#define FUNC_MACRO(Fp, Sp) \
    template void swap_gate(std::uint64_t, std::uint64_t, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
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
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template void swap_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
=======
#define FUNC_MACRO(Fp, Sp) \
    template void swap_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
>>>>>>> set-space
#undef FUNC_MACRO
}  // namespace scaluq::internal
