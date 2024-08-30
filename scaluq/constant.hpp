#pragma once

#include <array>
#include <numbers>

#include "types.hpp"

namespace scaluq {

#define DEF_MATH_CONSTANT(TRAIT, VALUE)                                                        \
    template <class T>                                                                         \
    inline constexpr auto TRAIT##_v = std::enable_if_t<std::is_floating_point_v<T>, T>(VALUE); \
    inline constexpr auto TRAIT = TRAIT##_v<double>

//! PI value
DEF_MATH_CONSTANT(PI, 3.141592653589793);

//! square root of 2
DEF_MATH_CONSTANT(SQRT2, 1.4142135623730950);

//! inverse square root of 2
DEF_MATH_CONSTANT(INVERSE_SQRT2, 0.7071067811865475);

//! cosine pi/8
DEF_MATH_CONSTANT(COSPI8, 0.923879532511287);

//! sine pi/8
DEF_MATH_CONSTANT(SINPI8, 0.382683432365090);

//! identity matrix
KOKKOS_INLINE_FUNCTION
matrix_2_2 I_GATE() { return {1, 0, 0, 1}; }
//! Pauli matrix X
KOKKOS_INLINE_FUNCTION
matrix_2_2 X_GATE() { return {0, 1, 1, 0}; }
//! Pauli matrix Y
KOKKOS_INLINE_FUNCTION
matrix_2_2 Y_GATE() { return {0, Complex(0, -1), Complex(0, 1), 0}; }
//! Pauli matrix Z
KOKKOS_INLINE_FUNCTION
matrix_2_2 Z_GATE() { return {1, 0, 0, -1}; }

//! list of Pauli matrix I,X,Y,Z
// std::array<matrix_2_2, 4> PAULI_MATRIX = {I_GATE, X_GATE, Y_GATE, Z_GATE};

//! S-gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 S_GATE_MATRIX() { return {1, 0, 0, Complex(0, 1)}; }
//! Sdag-gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 S_DAG_GATE_MATRIX() { return {1, 0, 0, Complex(0, -1)}; }
//! T-gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 T_GATE_MATRIX() {
    return {COSPI8 - Complex(0, SINPI8), 0., 0., COSPI8 + Complex(0, SINPI8) * SINPI8};
}
//! Tdag-gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 T_DAG_GATE_MATRIX() {
    return {COSPI8 + Complex(0, SINPI8), 0., 0., COSPI8 - Complex(0, SINPI8)};
}
//! Hadamard gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 HADAMARD_MATRIX() {
    return {0.7071067811865475, 0.7071067811865475, 0.7071067811865475, -0.7071067811865475};
}
//! square root of X gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 SQRT_X_GATE_MATRIX() {
    return {Complex(0.5, 0.5), Complex(0.5, -0.5), Complex(0.5, -0.5), Complex(0.5, 0.5)};
}
//! square root of Y gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 SQRT_Y_GATE_MATRIX() {
    return {Complex(0.5, 0.5), Complex(-0.5, -0.5), Complex(0.5, 0.5), Complex(0.5, 0.5)};
}
//! square root dagger of X gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 SQRT_X_DAG_GATE_MATRIX() {
    return {Complex(0.5, -0.5), Complex(0.5, 0.5), Complex(0.5, 0.5), Complex(0.5, -0.5)};
}
//! square root dagger of Y gate
KOKKOS_INLINE_FUNCTION
matrix_2_2 SQRT_Y_DAG_GATE_MATRIX() {
    return {Complex(0.5, -0.5), Complex(0.5, -0.5), Complex(-0.5, 0.5), Complex(0.5, -0.5)};
}
//! Projection to 0
KOKKOS_INLINE_FUNCTION
matrix_2_2 PROJ_0_MATRIX() { return {1, 0, 0, 0}; }
//! Projection to 1
KOKKOS_INLINE_FUNCTION
matrix_2_2 PROJ_1_MATRIX() { return {0, 0, 0, 1}; }
//! complex values for exp(j * i*pi/4 )
KOKKOS_INLINE_FUNCTION
array_4 PHASE_90ROT() { return {1., Complex(0, 1), -1, Complex(0, -1)}; }
//! complex values for exp(-j * i*pi/4 )
KOKKOS_INLINE_FUNCTION
array_4 PHASE_M90ROT() { return {1., Complex(0, -1), -1, Complex(0, 1)}; }
}  // namespace scaluq
