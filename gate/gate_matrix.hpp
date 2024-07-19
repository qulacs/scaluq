#pragma once

#include <ranges>
#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_one_qubit.hpp"
#include "gate_two_qubit.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
class OneQubitMatrixGateImpl : public GateBase {
    matrix_2_2 _matrix;

public:
    OneQubitMatrixGateImpl(UINT target_mask,
                           UINT control_mask,
                           const std::array<std::array<Complex, 2>, 2>& matrix)
        : GateBase(target_mask, control_mask) {
        _matrix.val[0][0] = matrix[0][0];
        _matrix.val[0][1] = matrix[0][1];
        _matrix.val[1][0] = matrix[1][0];
        _matrix.val[1][1] = matrix[1][1];
    }

    std::array<std::array<Complex, 2>, 2> matrix() const {
        return {_matrix.val[0][0], _matrix.val[0][1], _matrix.val[1][0], _matrix.val[1][1]};
    }

    Gate get_inverse() const override {
        return std::make_shared<OneQubitMatrixGateImpl>(
            _target_mask,
            _control_mask,
            std::array<std::array<Complex, 2>, 2>{Kokkos::conj(_matrix.val[0][0]),
                                                  Kokkos::conj(_matrix.val[1][0]),
                                                  Kokkos::conj(_matrix.val[0][1]),
                                                  Kokkos::conj(_matrix.val[1][1])});
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << this->_matrix.val[0][0], this->_matrix.val[0][1], this->_matrix.val[1][0],
            this->_matrix.val[1][1];
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        single_qubit_dense_matrix_gate(_target_mask, _control_mask, _matrix, state_vector);
    }
};

class TwoQubitMatrixGateImpl : public GateBase {
    matrix_4_4 _matrix;

public:
    TwoQubitMatrixGateImpl(UINT target_mask,
                           UINT control_mask,
                           const std::array<std::array<Complex, 4>, 4>& matrix)
        : GateBase(target_mask, control_mask) {
        for (UINT i : std::views::iota(0, 4)) {
            for (UINT j : std::views::iota(0, 4)) {
                _matrix.val[i][j] = matrix[i][j];
            }
        }
    }

    std::array<std::array<Complex, 4>, 4> matrix() const {
        std::array<std::array<Complex, 4>, 4> matrix;
        for (UINT i : std::views::iota(0, 4)) {
            for (UINT j : std::views::iota(0, 4)) {
                matrix[i][j] = _matrix.val[i][j];
            }
        }
        return matrix;
    }

    Gate get_inverse() const override {
        std::array<std::array<Complex, 4>, 4> matrix_dag;
        for (UINT i : std::views::iota(0, 4)) {
            for (UINT j : std::views::iota(0, 4)) {
                matrix_dag[i][j] = Kokkos::conj(_matrix.val[j][i]);
            }
        }
        return std::make_shared<TwoQubitMatrixGateImpl>(_target_mask, _control_mask, matrix_dag);
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(4, 4);
        mat << this->_matrix.val[0][0], this->_matrix.val[0][1], this->_matrix.val[0][2],
            this->_matrix.val[0][3], this->_matrix.val[1][0], this->_matrix.val[1][1],
            this->_matrix.val[1][2], this->_matrix.val[1][3], this->_matrix.val[2][0],
            this->_matrix.val[2][1], this->_matrix.val[2][2], this->_matrix.val[2][3],
            this->_matrix.val[3][0], this->_matrix.val[3][1], this->_matrix.val[3][2],
            this->_matrix.val[3][3];
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        double_qubit_dense_matrix_gate(_target_mask, _control_mask, _matrix, state_vector);
    }
};
}  // namespace internal

using OneQubitMatrixGate = internal::GatePtr<internal::OneQubitMatrixGateImpl>;
using TwoQubitMatrixGate = internal::GatePtr<internal::TwoQubitMatrixGateImpl>;
}  // namespace scaluq
