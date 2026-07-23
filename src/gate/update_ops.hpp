#pragma once

#include <bit>
#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/state/density_matrix.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>
#include <scaluq/types.hpp>

namespace scaluq {
namespace internal {

template <typename State>
#if defined(KOKKOS_ARCH_AVX2)
inline constexpr bool supports_gate_simd = [] {
    if constexpr ((State::space == ExecutionSpace::Host ||
                   State::space == ExecutionSpace::HostSerial) &&
                  (State::prec == Precision::F64 || State::prec == Precision::F32)) {
        return SimdComplex<State::prec>::complex_lanes > 0;
    }
    return false;
}();
#else
inline constexpr bool supports_gate_simd = false;
#endif

template <typename SimdType, typename State>
    requires supports_gate_simd<State>
inline bool can_use_gate_simd_for(std::uint64_t skip_mask, const State& state) {
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const std::uint64_t span = state.dim() >> std::popcount(skip_mask);
    return span >= complex_lanes && (skip_mask & (complex_lanes - 1)) == 0;
}

template <typename State>
    requires supports_gate_simd<State>
inline bool can_use_gate_simd(std::uint64_t skip_mask, const State& state) {
    return can_use_gate_simd_for<SimdComplex<State::prec>>(skip_mask, state);
}

template <CoefKind Kind = CoefKind::General, UpdatableStateVector State>
void zero_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   Complex<State::prec> matrix,
                                   State& state);

template <CoefKind M00 = CoefKind::General,
          CoefKind M01 = CoefKind::General,
          CoefKind M10 = CoefKind::General,
          CoefKind M11 = CoefKind::General,
          UpdatableStateVector State>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<State::prec>& matrix,
                                  State& state);

// clang-format off
template <CoefKind M00 = CoefKind::General, CoefKind M01 = CoefKind::General, CoefKind M02 = CoefKind::General, CoefKind M03 = CoefKind::General,
          CoefKind M10 = CoefKind::General, CoefKind M11 = CoefKind::General, CoefKind M12 = CoefKind::General, CoefKind M13 = CoefKind::General,
          CoefKind M20 = CoefKind::General, CoefKind M21 = CoefKind::General, CoefKind M22 = CoefKind::General, CoefKind M23 = CoefKind::General,
          CoefKind M30 = CoefKind::General, CoefKind M31 = CoefKind::General, CoefKind M32 = CoefKind::General, CoefKind M33 = CoefKind::General,
          UpdatableStateVector State>
// clang-format on
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<State::prec>& matrix,
                                  State& state);

template <UpdatableStateVector State>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<State::prec, State::space>& matrix,
                             State& state);

template <UpdatableStateVector State>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<State::prec, State::space>& mat,
                        State& state);

template <CoefKind Kind = CoefKind::General, UpdatableStateVector State>
inline void one_target_phase_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  Complex<State::prec> phase,
                                  State& state) {
    zero_target_dense_matrix_gate<Kind>(
        control_mask | target_mask, control_value_mask | target_mask, phase, state);
}

template <UpdatableStateVector State>
inline void global_phase_gate(std::uint64_t,
                              std::uint64_t control_mask,
                              std::uint64_t control_value_mask,
                              Float<State::prec> angle,
                              State& state) {
    zero_target_dense_matrix_gate(control_mask,
                                  control_value_mask,
                                  internal::polar<State::prec>(Float<State::prec>{1}, angle),
                                  state);
}

template <UpdatableStateVector State>
inline void i_gate(std::uint64_t, std::uint64_t, std::uint64_t, State&) {}

template <UpdatableStateVector State>
inline void x_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_dense_matrix_gate<CoefKind::Zero, CoefKind::One, CoefKind::One, CoefKind::Zero>(
        target_mask, control_mask, control_value_mask, X_GATE<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void y_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_dense_matrix_gate<CoefKind::Zero, CoefKind::Imag, CoefKind::Imag, CoefKind::Zero>(
        target_mask, control_mask, control_value_mask, Y_GATE<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void z_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_phase_gate<CoefKind::Real>(
        target_mask, control_mask, control_value_mask, Complex<State::prec>(-1, 0), state);
}

template <UpdatableStateVector State>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_dense_matrix_gate<CoefKind::Real, CoefKind::Real, CoefKind::Real, CoefKind::Real>(
        target_mask, control_mask, control_value_mask, HADAMARD_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_phase_gate<CoefKind::Imag>(
        target_mask, control_mask, control_value_mask, Complex<State::prec>(0, 1), state);
}

template <UpdatableStateVector State>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      State& state) {
    one_target_phase_gate<CoefKind::Imag>(
        target_mask, control_mask, control_value_mask, Complex<State::prec>(0, -1), state);
}

template <UpdatableStateVector State>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<State::prec>(INVERSE_SQRT2(), INVERSE_SQRT2()),
                          state);
}

template <UpdatableStateVector State>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      State& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<State::prec>(INVERSE_SQRT2(), -INVERSE_SQRT2()),
                          state);
}

template <UpdatableStateVector State>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       State& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_GATE_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          State& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 SQRT_X_DAG_GATE_MATRIX<State::prec>(),
                                 state);
}

template <UpdatableStateVector State>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       State& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_GATE_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          State& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 SQRT_Y_DAG_GATE_MATRIX<State::prec>(),
                                 state);
}

template <UpdatableStateVector State>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    State& state) {
    one_target_dense_matrix_gate<CoefKind::One, CoefKind::Zero, CoefKind::Zero, CoefKind::Zero>(
        target_mask, control_mask, control_value_mask, PROJ_0_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    State& state) {
    one_target_dense_matrix_gate<CoefKind::Zero, CoefKind::Zero, CoefKind::Zero, CoefKind::One>(
        target_mask, control_mask, control_value_mask, PROJ_1_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void rx_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<State::prec> angle,
                    State& state) {
    using ComplexType = Complex<State::prec>;
    const Float<State::prec> cosval = internal::cos(angle / Float<State::prec>{2});
    const Float<State::prec> sinval = internal::sin(angle / Float<State::prec>{2});
    const Matrix2x2<State::prec> matrix = {
        {{{cosval, ComplexType(0, -sinval)}}, {{ComplexType(0, -sinval), cosval}}}};
    one_target_dense_matrix_gate<CoefKind::Real, CoefKind::Imag, CoefKind::Imag, CoefKind::Real>(
        target_mask, control_mask, control_value_mask, matrix, state);
}
template <Precision Prec, ExecutionSpace Space>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states);

template <UpdatableStateVector State>
inline void ry_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<State::prec> angle,
                    State& state) {
    const Float<State::prec> cosval = internal::cos(angle / Float<State::prec>{2});
    const Float<State::prec> sinval = internal::sin(angle / Float<State::prec>{2});
    const Matrix2x2<State::prec> matrix = {{{{cosval, -sinval}}, {{sinval, cosval}}}};
    one_target_dense_matrix_gate<CoefKind::Real, CoefKind::Real, CoefKind::Real, CoefKind::Real>(
        target_mask, control_mask, control_value_mask, matrix, state);
}
template <Precision Prec, ExecutionSpace Space>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states);

template <UpdatableStateVector State>
inline void rz_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<State::prec> angle,
                    State& state) {
    using ComplexType = Complex<State::prec>;
    const Float<State::prec> cosval = internal::cos(angle / Float<State::prec>{2});
    const Float<State::prec> sinval = internal::sin(angle / Float<State::prec>{2});
    const Matrix2x2<State::prec> matrix = {{{{ComplexType(cosval, -sinval), ComplexType{}}},
          {{ComplexType{}, ComplexType(cosval, sinval)}}}};
    one_target_dense_matrix_gate<CoefKind::General,
                                 CoefKind::Zero,
                                 CoefKind::Zero,
                                 CoefKind::General>(
        target_mask, control_mask, control_value_mask, matrix, state);
}
template <Precision Prec, ExecutionSpace Space>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states);

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> get_IBMQ_matrix(Float<Prec> theta,
                                                       Float<Prec> phi,
                                                       Float<Prec> lambda) {
    const Complex<Prec> exp_val1 = internal::exp(Complex<Prec>(0, phi));
    const Complex<Prec> exp_val2 = internal::exp(Complex<Prec>(0, lambda));
    const Complex<Prec> cos_val = internal::cos(theta / Float<Prec>{2});
    const Complex<Prec> sin_val = internal::sin(theta / Float<Prec>{2});
    return {
        {{{cos_val, -exp_val2 * sin_val}}, {{exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val}}}};
}

template <UpdatableStateVector State>
inline void u1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<State::prec> lambda,
                    State& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          internal::exp(Complex<State::prec>(0, lambda)),
                          state);
}

template <UpdatableStateVector State>
inline void u2_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<State::prec> phi,
                    Float<State::prec> lambda,
                    State& state) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        control_value_mask,
        get_IBMQ_matrix<State::prec>(
            static_cast<Float<State::prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        state);
}

template <UpdatableStateVector State>
inline void u3_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<State::prec> theta,
                    Float<State::prec> phi,
                    Float<State::prec> lambda,
                    State& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 get_IBMQ_matrix<State::prec>(theta, phi, lambda),
                                 state);
}

template <UpdatableStateVector State>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               State& state);

template <UpdatableStateVector State>
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              State& state);
template <UpdatableStateVector State>
void permutation_gate(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& swap_schedule,
                      State& state);

// DensityMatrix overloads

template <Precision Prec, ExecutionSpace Space>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
inline void u1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<Prec> lambda,
                    DensityMatrix<Prec, Space>& dm) {
    const DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(1, 0),
                                          internal::exp(Complex<Prec>(0, lambda))};
    one_target_diagonal_matrix_gate(target_mask, control_mask, control_value_mask, diag, dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void u2_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<Prec> phi,
                    Float<Prec> lambda,
                    DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        control_value_mask,
        get_IBMQ_matrix<Prec>(static_cast<Float<Prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void u3_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    Float<Prec> theta,
                    Float<Prec> phi,
                    Float<Prec> lambda,
                    DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 get_IBMQ_matrix<Prec>(theta, phi, lambda),
                                 dm);
}

template <Precision Prec, ExecutionSpace Space>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              DensityMatrix<Prec, Space>& dm);

template <Precision Prec, ExecutionSpace Space>
inline void i_gate(std::uint64_t, std::uint64_t, std::uint64_t, DensityMatrix<Prec, Space>&) {}

template <Precision Prec, ExecutionSpace Space>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, HADAMARD_MATRIX<Prec>(), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   DensityMatrix<Prec, Space>& dm) {
    one_target_phase_gate(target_mask, control_mask, control_value_mask, Complex<Prec>(0, 1), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      DensityMatrix<Prec, Space>& dm) {
    one_target_phase_gate(target_mask, control_mask, control_value_mask, Complex<Prec>(0, -1), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   DensityMatrix<Prec, Space>& dm) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<Prec>(INVERSE_SQRT2(), INVERSE_SQRT2()),
                          dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      DensityMatrix<Prec, Space>& dm) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<Prec>(INVERSE_SQRT2(), -INVERSE_SQRT2()),
                          dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_GATE_MATRIX<Prec>(), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_DAG_GATE_MATRIX<Prec>(), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_GATE_MATRIX<Prec>(), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_DAG_GATE_MATRIX<Prec>(), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_0_MATRIX<Prec>(), dm);
}

template <Precision Prec, ExecutionSpace Space>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    DensityMatrix<Prec, Space>& dm) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_1_MATRIX<Prec>(), dm);
}

}  // namespace internal
}  // namespace scaluq
