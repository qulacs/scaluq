#pragma once

#include <array>
#include <numbers>

#include "../types.hpp"

namespace qulacs {
//! PI value
constexpr double PI = std::numbers::pi;

//! square root of 2
constexpr double SQRT2 = std::numbers::sqrt2;

//! inverse square root of 2
constexpr double INVERSE_SQRT2 = 0.707106781186547;

//! cosine pi/8
constexpr double COSPI8 = 0.923879532511287;

//! sine pi/8
constexpr double SINPI8 = 0.382683432365090;

//! identity matrix
constexpr std::array<Complex, 4> I_GATE = {1, 0, 0, 1};
//! Pauli matrix X
constexpr std::array<Complex, 4> X_GATE = {0, 1, 1, 0};
//! Pauli matrix Y
constexpr std::array<Complex, 4> Y_GATE = {0, Complex(0, -1), Complex(0, 1), 0};
//! Pauli matrix Z
constexpr std::array<Complex, 4> Z_GATE = {1, 0, 0, -1};
//! list of Pauli matrix I,X,Y,Z
constexpr std::array<std::array<Complex, 4>, 4> PAULI_MATRIX = {I_GATE, X_GATE, Y_GATE, Z_GATE};
//! S-gate
constexpr std::array<Complex, 4> S_GATE_MATRIX = {1, 0, 0, Complex(0, 1)};
//! Sdag-gate
constexpr std::array<Complex, 4> S_DAG_GATE_MATRIX = {1, 0, 0, Complex(0, -1)};
//! T-gate
constexpr std::array<Complex, 4> T_GATE_MATRIX = {
    COSPI8 - Complex(0, 1) * SINPI8, 0., 0., COSPI8 + Complex(0, 1) * SINPI8};
//! Tdag-gate
constexpr std::array<Complex, 4> T_DAG_GATE_MATRIX = {
    COSPI8 + Complex(0, 1) * SINPI8, 0., 0., COSPI8 - Complex(0, 1) * SINPI8};
//! Hadamard gate
constexpr std::array<Complex, 4> HADAMARD_MATRIX = {
    1. / SQRT2, 1. / SQRT2, 1. / SQRT2, -1. / SQRT2};
//! square root of X gate
constexpr std::array<Complex, 4> SQRT_X_GATE_MATRIX = {
    Complex(0.5, 0.5), Complex(0.5, -0.5), Complex(0.5, -0.5), Complex(0.5, 0.5)};
//! square root of Y gate
constexpr std::array<Complex, 4> SQRT_Y_GATE_MATRIX = {
    0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i};
//! square root dagger of X gate
constexpr std::array<Complex, 4> SQRT_X_DAG_GATE_MATRIX = {
    0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i};
//! square root dagger of Y gate
constexpr std::array<Complex, 4> SQRT_Y_DAG_GATE_MATRIX = {
    0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i};
//! Projection to 0
constexpr std::array<Complex, 4> PROJ_0_MATRIX = {1, 0, 0, 0};
//! Projection to 1
constexpr std::array<Complex, 4> PROJ_1_MATRIX = {0, 0, 0, 1};
//! complex values for exp(j * i*pi/4 )
constexpr std::array<Complex, 4> PHASE_90ROT = {1., 1.i, -1, -1.i};
//! complex values for exp(-j * i*pi/4 )
constexpr std::array<Complex, 4> PHASE_M90ROT = {1., -1.i, -1, 1.i};
}  // namespace qulacs
