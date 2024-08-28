#pragma once

#include <ranges>
#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_standard.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
class OneTargetMatrixGateImpl : public GateBase {
    matrix_2_2 _matrix;

public:
    OneTargetMatrixGateImpl(UINT target_mask,
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
        return std::make_shared<const OneTargetMatrixGateImpl>(
            _target_mask,
            _control_mask,
            std::array<std::array<Complex, 2>, 2>{Kokkos::conj(_matrix.val[0][0]),
                                                  Kokkos::conj(_matrix.val[1][0]),
                                                  Kokkos::conj(_matrix.val[0][1]),
                                                  Kokkos::conj(_matrix.val[1][1])});
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << this->_matrix.val[0][0], this->_matrix.val[0][1], this->_matrix.val[1][0],
            this->_matrix.val[1][1];
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        one_target_dense_matrix_gate(_target_mask, _control_mask, _matrix, state_vector);
    }
};

class TwoTargetMatrixGateImpl : public GateBase {
    matrix_4_4 _matrix;

public:
    TwoTargetMatrixGateImpl(UINT target_mask,
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
        return std::make_shared<const TwoTargetMatrixGateImpl>(
            _target_mask, _control_mask, matrix_dag);
    }
    ComplexMatrix get_matrix() const override {
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
        two_target_dense_matrix_gate(_target_mask, _control_mask, _matrix, state_vector);
    }
};

class SparseMatrixGateImpl : public GateBase {
    SparseMatrix _matrix;
    SparseMatrixGateImpl(UINT target_mask, UINT control_mask, const SparseComplexMatrix& mat)
        : GateBase(target_mask, control_mask), _matrix(SparseMatrix(mat)) {}

    Gate get_inverse() const override {
        throw std::logic_error("Error: SparseMatrixGateImpl::get_inverse(): Not implemented.");
    }

    Matrix get_matrix_internal() const {
        Matrix ret("return matrix", _matrix._row, _matrix._col);
        auto vec = _matrix._values;
        for (int i = 0; i < vec.size(); i++) {
            ret(vec[i].r, vec[i].c) = vec[i].val;
        }
        return ret;
    }

    ComplexMatrix get_matrix() const override {
        ComplexMatrix ret(_matrix._row, _matrix._col);
        auto vec = _matrix._values;
        for (int i = 0; i < vec.size(); i++) {
            ret(vec[i].r, vec[i].c) = vec[i].val;
        }
        return ret;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sparse_matrix_gate(_target_mask, _control_mask, _matrix, state_vector);
    }
};

class DenseMatrixGateImpl : public GateBase {
    Matrix _matrix;
    bool _is_unitary;

    DenseMatrixGateImpl(UINT target_mask,
                        UINT control_mask,
                        const ComplexMatrix& mat,
                        bool is_unitary = false)
        : GateBase(target_mask, control_mask),
          _is_unitary(is_unitary),
          _matrix(convert_external_matrix_to_internal_matrix(mat)) {}

    Gate get_inverse() const override {
        ComplexMatrix mat_eigen = convert_internal_matrix_to_external_matrix(_matrix);
        ComplexMatrix inv_eigen;
        if (_is_unitary) {
            inv_eigen = mat_eigen.adjoint();
        } else {
            inv_eigen = mat_eigen.inverse();
        }
        return std::make_shared<const DenseMatrixGateImpl>(
            _target_mask, _control_mask, inv_eigen, _is_unitary);
    }

    Matrix get_matrix_internal() const { return _matrix; }

    ComplexMatrix get_matrix() const override {
        return convert_internal_matrix_to_external_matrix(_matrix);
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        dense_matrix_gate(_target_mask, _control_mask, _matrix, state_vector);
    }
};

}  // namespace internal

using OneTargetMatrixGate = internal::GatePtr<internal::OneTargetMatrixGateImpl>;
using TwoTargetMatrixGate = internal::GatePtr<internal::TwoTargetMatrixGateImpl>;
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl>;
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl>;
}  // namespace scaluq
