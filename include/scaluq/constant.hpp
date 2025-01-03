#pragma once

#include <array>
#include <numbers>

#include "types.hpp"

namespace scaluq {

namespace internal {

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
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> I_GATE() {
    return {1, 0, 0, 1};
}
//! Pauli matrix X
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> X_GATE() {
    return {0, 1, 1, 0};
}
//! Pauli matrix Y
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> Y_GATE() {
    return {0, Complex<Fp>(0, -1), Complex<Fp>(0, 1), 0};
}
//! Pauli matrix Z
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> Z_GATE() {
    return {1, 0, 0, -1};
}

//! list of Pauli matrix I,X,Y,Z
// std::array<Matrix2x2, 4> PAULI_MATRIX = {I_GATE, X_GATE,
// Y_GATE, Z_GATE};

//! S-gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> S_GATE_MATRIX() {
    return {1, 0, 0, Complex<Fp>(0, 1)};
}
//! Sdag-gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> S_DAG_GATE_MATRIX() {
    return {1, 0, 0, Complex<Fp>(0, -1)};
}
//! T-gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> T_GATE_MATRIX() {
    return {1, 0, 0, Complex<Fp>(INVERSE_SQRT2(), INVERSE_SQRT2())};
}
//! Tdag-gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> T_DAG_GATE_MATRIX() {
    return {1, 0, 0, Complex<Fp>(INVERSE_SQRT2(), -INVERSE_SQRT2())};
}
//! Hadamard gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> HADAMARD_MATRIX() {
    return {INVERSE_SQRT2(), INVERSE_SQRT2(), INVERSE_SQRT2(), -INVERSE_SQRT2()};
}
//! square root of X gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> SQRT_X_GATE_MATRIX() {
    return {Complex<Fp>(0.5, 0.5),
            Complex<Fp>(0.5, -0.5),
            Complex<Fp>(0.5, -0.5),
            Complex<Fp>(0.5, 0.5)};
}
//! square root of Y gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> SQRT_Y_GATE_MATRIX() {
    return {Complex<Fp>(0.5, 0.5),
            Complex<Fp>(-0.5, -0.5),
            Complex<Fp>(0.5, 0.5),
            Complex<Fp>(0.5, 0.5)};
}
//! square root dagger of X gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> SQRT_X_DAG_GATE_MATRIX() {
    return {Complex<Fp>(0.5, -0.5),
            Complex<Fp>(0.5, 0.5),
            Complex<Fp>(0.5, 0.5),
            Complex<Fp>(0.5, -0.5)};
}
//! square root dagger of Y gate
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> SQRT_Y_DAG_GATE_MATRIX() {
    return {Complex<Fp>(0.5, -0.5),
            Complex<Fp>(0.5, -0.5),
            Complex<Fp>(-0.5, 0.5),
            Complex<Fp>(0.5, -0.5)};
}
//! Projection to 0
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> PROJ_0_MATRIX() {
    return {1, 0, 0, 0};
}
//! Projection to 1
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Matrix2x2<Fp> PROJ_1_MATRIX() {
    return {0, 0, 0, 1};
}
//! complex values for exp(j * i*pi/4 )
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Complex<Fp>, 4> PHASE_90ROT() {
    return {1., Complex<Fp>(0, 1), -1, Complex<Fp>(0, -1)};
}
//! complex values for exp(-j * i*pi/4 )
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Complex<Fp>, 4> PHASE_M90ROT() {
    return {1., Complex<Fp>(0, -1), -1, Complex<Fp>(0, 1)};
}
}  // namespace internal
}  // namespace scaluq
