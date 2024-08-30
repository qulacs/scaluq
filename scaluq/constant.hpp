#pragma once

#include <array>
#include <numbers>

#include "types.hpp"

namespace scaluq {

namespace internal {

#define DEF_MATH_CONSTANT(TRAIT, VALUE)                                                        \
    template <class T>                                                                         \
    inline constexpr auto TRAIT##_v = std::enable_if_t<std::is_floating_point_v<T>, T>(VALUE); \
    inline constexpr auto TRAIT = TRAIT##_v<double>

//! inverse square root of 2
KOKKOS_INLINE_FUNCTION
double INVERSE_SQRT2() { return Kokkos::numbers::sqrt2 / 2; }

//! cosine pi/8
KOKKOS_INLINE_FUNCTION
double COSPI8() { return 0.923879532511287; }

//! sine pi/8
KOKKOS_INLINE_FUNCTION
double SINPI8() { return 0.382683432365090; }

//! identity matrix
KOKKOS_INLINE_FUNCTION
Matrix2x2 I_GATE() { return {1, 0, 0, 1}; }
//! Pauli matrix X
KOKKOS_INLINE_FUNCTION
Matrix2x2 X_GATE() { return {0, 1, 1, 0}; }
//! Pauli matrix Y
KOKKOS_INLINE_FUNCTION
Matrix2x2 Y_GATE() { return {0, Complex(0, -1), Complex(0, 1), 0}; }
//! Pauli matrix Z
KOKKOS_INLINE_FUNCTION
Matrix2x2 Z_GATE() { return {1, 0, 0, -1}; }

//! list of Pauli matrix I,X,Y,Z
// std::array<Matrix2x2, 4> PAULI_MATRIX = {I_GATE, X_GATE,
// Y_GATE, Z_GATE};

//! S-gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 S_GATE_MATRIX() { return {1, 0, 0, Complex(0, 1)}; }
//! Sdag-gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 S_DAG_GATE_MATRIX() { return {1, 0, 0, Complex(0, -1)}; }
//! T-gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 T_GATE_MATRIX() { return {1, 0, 0, Complex(INVERSE_SQRT2(), INVERSE_SQRT2())}; }
//! Tdag-gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 T_DAG_GATE_MATRIX() { return {1, 0, 0, Complex(INVERSE_SQRT2(), -INVERSE_SQRT2())}; }
//! Hadamard gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 HADAMARD_MATRIX() {
    return {INVERSE_SQRT2(), INVERSE_SQRT2(), INVERSE_SQRT2(), -INVERSE_SQRT2()};
}
//! square root of X gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 SQRT_X_GATE_MATRIX() {
    return {Complex(0.5, 0.5), Complex(0.5, -0.5), Complex(0.5, -0.5), Complex(0.5, 0.5)};
}
//! square root of Y gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 SQRT_Y_GATE_MATRIX() {
    return {Complex(0.5, 0.5), Complex(-0.5, -0.5), Complex(0.5, 0.5), Complex(0.5, 0.5)};
}
//! square root dagger of X gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 SQRT_X_DAG_GATE_MATRIX() {
    return {Complex(0.5, -0.5), Complex(0.5, 0.5), Complex(0.5, 0.5), Complex(0.5, -0.5)};
}
//! square root dagger of Y gate
KOKKOS_INLINE_FUNCTION
Matrix2x2 SQRT_Y_DAG_GATE_MATRIX() {
    return {Complex(0.5, -0.5), Complex(0.5, -0.5), Complex(-0.5, 0.5), Complex(0.5, -0.5)};
}
//! Projection to 0
KOKKOS_INLINE_FUNCTION
Matrix2x2 PROJ_0_MATRIX() { return {1, 0, 0, 0}; }
//! Projection to 1
KOKKOS_INLINE_FUNCTION
Matrix2x2 PROJ_1_MATRIX() { return {0, 0, 0, 1}; }
//! complex values for exp(j * i*pi/4 )
KOKKOS_INLINE_FUNCTION
Kokkos::Array<Complex, 4> PHASE_90ROT() { return {1., Complex(0, 1), -1, Complex(0, -1)}; }
//! complex values for exp(-j * i*pi/4 )
KOKKOS_INLINE_FUNCTION
Kokkos::Array<Complex, 4> PHASE_M90ROT() { return {1., Complex(0, -1), -1, Complex(0, 1)}; }
}  // namespace internal
}  // namespace scaluq
