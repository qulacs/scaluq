#pragma once

#include <array>
#include <numbers>

#include "types.hpp"

namespace scaluq {

namespace internal {

//! inverse square root of 2
KOKKOS_INLINE_FUNCTION
constexpr double INVERSE_SQRT2() { return Kokkos::numbers::sqrt2 / 2; }

//! cosine pi/8
KOKKOS_INLINE_FUNCTION
constexpr double COSPI8() { return 0.923879532511287; }

//! sine pi/8
KOKKOS_INLINE_FUNCTION
constexpr double SINPI8() { return 0.382683432365090; }

//! identity matrix
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> I_GATE() {
    return {1, 0, 0, 1};
}
//! Pauli matrix X
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> X_GATE() {
    return {0, 1, 1, 0};
}
//! Pauli matrix Y
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> Y_GATE() {
    return {0, Complex<Prec>(0, -1), Complex<Prec>(0, 1), 0};
}
//! Pauli matrix Z
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> Z_GATE() {
    return {1, 0, 0, -1};
}

//! list of Pauli matrix I,X,Y,Z
// std::array<Matrix2x2, 4> PAULI_MATRIX = {I_GATE, X_GATE,
// Y_GATE, Z_GATE};

//! S-gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> S_GATE_MATRIX() {
    return {1, 0, 0, Complex<Prec>(0, 1)};
}
//! Sdag-gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> S_DAG_GATE_MATRIX() {
    return {1, 0, 0, Complex<Prec>(0, -1)};
}
//! T-gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> T_GATE_MATRIX() {
    return {1, 0, 0, Complex<Prec>(INVERSE_SQRT2(), INVERSE_SQRT2())};
}
//! Tdag-gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> T_DAG_GATE_MATRIX() {
    return {1, 0, 0, Complex<Prec>(INVERSE_SQRT2(), -INVERSE_SQRT2())};
}
//! Hadamard gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> HADAMARD_MATRIX() {
    Float<Prec> ISQRT2 = static_cast<Float<Prec>>(INVERSE_SQRT2());
    return {ISQRT2, ISQRT2, ISQRT2, -ISQRT2};
}
//! square root of X gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> SQRT_X_GATE_MATRIX() {
    return {Complex<Prec>(.5, .5),
            Complex<Prec>(.5, -.5),
            Complex<Prec>(.5, -.5),
            Complex<Prec>(.5, .5)};
}
//! square root of Y gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> SQRT_Y_GATE_MATRIX() {
    return {Complex<Prec>(.5, .5),
            Complex<Prec>(-.5, -.5),
            Complex<Prec>(.5, .5),
            Complex<Prec>(.5, .5)};
}
//! square root dagger of X gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> SQRT_X_DAG_GATE_MATRIX() {
    return {Complex<Prec>(.5, -.5),
            Complex<Prec>(.5, .5),
            Complex<Prec>(.5, .5),
            Complex<Prec>(.5, -.5)};
}
//! square root dagger of Y gate
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> SQRT_Y_DAG_GATE_MATRIX() {
    return {Complex<Prec>(.5, -.5),
            Complex<Prec>(.5, -.5),
            Complex<Prec>(-.5, .5),
            Complex<Prec>(.5, -.5)};
}
//! Projection to 0
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> PROJ_0_MATRIX() {
    return {1, 0, 0, 0};
}
//! Projection to 1
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Matrix2x2<Prec> PROJ_1_MATRIX() {
    return {0, 0, 0, 1};
}
//! complex values for exp(j * i*pi/4 )
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Complex<Prec>, 4> PHASE_90ROT() {
    return {1, Complex<Prec>(0, 1), -1, Complex<Prec>(0, -1)};
}
//! complex values for exp(-j * i*pi/4 )
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Kokkos::Array<Complex<Prec>, 4> PHASE_M90ROT() {
    return {1, Complex<Prec>(0, -1), -1, Complex<Prec>(0, 1)};
}
}  // namespace internal
}  // namespace scaluq
