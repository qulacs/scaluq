#include "constant.hpp"

// elementary set of gates
const Complex PAULI_MATRIX[4][4] = {
    {1, 0, 0, 1}, {0, 1, 1, 0}, {0, -1.i, 1.i, 0}, {1, 0, 0, -1}};

const Complex S_GATE_MATRIX[4] = {1, 0, 0, 1.i};

const Complex S_DAG_GATE_MATRIX[4] = {1, 0, 0, -1.i};

const Complex T_GATE_MATRIX[4] = {
    COSPI8 - 1.i * SINPI8, 0., 0., COSPI8 + 1.i * SINPI8};

const Complex T_DAG_GATE_MATRIX[4] = {
    COSPI8 + 1.i * SINPI8, 0., 0., COSPI8 - 1.i * SINPI8};

const Complex HADAMARD_MATRIX[4] = {
    1. / SQRT2, 1. / SQRT2, 1. / SQRT2, -1. / SQRT2};

const Complex SQRT_X_GATE_MATRIX[4] = {
    0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i};

const Complex SQRT_Y_GATE_MATRIX[4] = {
    0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i};

const Complex SQRT_X_DAG_GATE_MATRIX[4] = {
    0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i};

const Complex SQRT_Y_DAG_GATE_MATRIX[4] = {
    0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i};

const Complex PROJ_0_MATRIX[4] = {1, 0, 0, 0};

const Complex PROJ_1_MATRIX[4] = {0, 0, 0, 1};

const Complex PHASE_90ROT[4] = {1., 1.i, -1, -1.i};
const Complex PHASE_M90ROT[4] = {1., -1.i, -1, 1.i};
