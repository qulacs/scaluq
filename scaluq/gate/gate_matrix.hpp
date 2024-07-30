#pragma once

#include <Eigen/Dense>
#include <ranges>
#include <vector>

#include "../info/qubit_info.hpp"
#include "../util/utility.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "constant.hpp"
#include "gate.hpp"
#include "gate_one_qubit.hpp"
#include "gate_two_qubit.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {

class OneQubitMatrixGateImpl : public OneQubitGateBase {
    matrix_2_2 _matrix;
    bool _is_unitary;

public:
    OneQubitMatrixGateImpl(UINT target,
                           const std::array<std::array<Complex, 2>, 2>& matrix,
                           bool is_unitary = false)
        : OneQubitGateBase(target), _is_unitary(is_unitary) {
        _matrix.val[0][0] = matrix[0][0];
        _matrix.val[0][1] = matrix[0][1];
        _matrix.val[1][0] = matrix[1][0];
        _matrix.val[1][1] = matrix[1][1];
    }

    std::array<std::array<Complex, 2>, 2> matrix() const {
        return {_matrix.val[0][0], _matrix.val[0][1], _matrix.val[1][0], _matrix.val[1][1]};
    }

    Gate get_inverse() const override {
        if (_is_unitary) {
            return std::make_shared<const OneQubitMatrixGateImpl>(
                _target,
                std::array<std::array<Complex, 2>, 2>{Kokkos::conj(_matrix.val[0][0]),
                                                      Kokkos::conj(_matrix.val[1][0]),
                                                      Kokkos::conj(_matrix.val[0][1]),
                                                      Kokkos::conj(_matrix.val[1][1])},
                _is_unitary);
        }
        Complex det = _matrix.val[0][0] * _matrix.val[1][1] - _matrix.val[0][1] * _matrix.val[1][0];
        if (std::abs((StdComplex)det) < 1e-10) {
            throw std::runtime_error(
                "OneQubitMatrixGateImpl::get_inverse(): Matrix is not invertible.");
        }
        std::array<std::array<Complex, 2>, 2> inv;
        inv[0][0] = _matrix.val[1][1] / det;
        inv[0][1] = -_matrix.val[0][1] / det;
        inv[1][0] = -_matrix.val[1][0] / det;
        inv[1][1] = _matrix.val[0][0] / det;

        return std::make_shared<const OneQubitMatrixGateImpl>(_target, inv, _is_unitary);
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << this->_matrix.val[0][0], this->_matrix.val[0][1], this->_matrix.val[1][0],
            this->_matrix.val[1][1];
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        single_qubit_dense_matrix_gate(_target, _matrix, state_vector);
    }
};

class TwoQubitMatrixGateImpl : public TwoQubitGateBase {
    matrix_4_4 _matrix;
    bool _is_unitary;

public:
    TwoQubitMatrixGateImpl(UINT target1,
                           UINT target2,
                           const std::array<std::array<Complex, 4>, 4>& matrix,
                           bool is_unitary = false)
        : TwoQubitGateBase(target1, target2), _is_unitary(is_unitary) {
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
        if (_is_unitary) {
            std::array<std::array<Complex, 4>, 4> matrix_dag;
            for (UINT i : std::views::iota(0, 4)) {
                for (UINT j : std::views::iota(0, 4)) {
                    matrix_dag[i][j] = Kokkos::conj(_matrix.val[j][i]);
                }
            }
            return std::make_shared<const TwoQubitMatrixGateImpl>(
                _target1, _target2, matrix_dag, _is_unitary);
        }
        ComplexMatrix mat_eigen(4, 4);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat_eigen(i, j) = _matrix.val[i][j];
            }
        }
        ComplexMatrix inv_eigen = mat_eigen.inverse();
        std::array<std::array<Complex, 4>, 4> mat;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat[i][j] = inv_eigen(i, j);
            }
        }
        return std::make_shared<const TwoQubitMatrixGateImpl>(_target1, _target2, mat, _is_unitary);
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
        check_qubit_within_bounds(state_vector, this->_target1);
        check_qubit_within_bounds(state_vector, this->_target2);
        double_qubit_dense_matrix_gate(_target1, _target2, _matrix, state_vector);
    }
};

class SparseMatrixGateImpl : public GateBase {
    CrsMatrix _matrix;
    std::vector<TargetQubitInfo> target_qubit_info_list;
    std::vector<ControlQubitInfo> control_qubit_info_list;

public:
    SparseMatrixGateImpl(SparseComplexMatrix matrix,
                         const std::vector<UINT>& target_qubit_index_list,
                         const std::vector<UINT>& control_qubit_index_list)
        : GateBase() {
        _matrix = std::move(convert_external_sparse_to_internal_sparse(matrix));
        for (UINT i = 0; i < target_qubit_index_list.size(); i++) {
            target_qubit_info_list.push_back(TargetQubitInfo(target_qubit_index_list[i]));
        }
        for (UINT i = 0; i < control_qubit_index_list.size(); i++) {
            control_qubit_info_list.push_back(ControlQubitInfo(control_qubit_index_list[i], 1));
        }
    }

    std::vector<UINT> get_target_qubit_list() const override {
        std::vector<UINT> target_qubit_list;
        for (const auto& target : target_qubit_info_list) {
            target_qubit_list.push_back(target.index());
        }
        return target_qubit_list;
    }
    std::vector<UINT> get_control_qubit_list() const override {
        std::vector<UINT> control_qubit_list;
        for (const auto& control : control_qubit_info_list) {
            control_qubit_list.push_back(control.index());
        }
        return control_qubit_list;
    }

    Gate get_inverse() const override {
        throw std::logic_error("Error: SparseMatrixGateImpl::get_inverse(): Not implemented.");
    }

    std::optional<Matrix> get_matrix_internal() const {
        Matrix mat("mat", _matrix.numRows(), _matrix.numCols());
        Kokkos::parallel_for(
            _matrix.numRows(), KOKKOS_LAMBDA(const auto i) {
                for (auto idx = _matrix.graph.row_map(i); idx < _matrix.graph.row_map(i + 1);
                     ++idx) {
                    auto j = _matrix.graph.entries(idx);
                    mat(i, j) = _matrix.values(idx);
                }
            });
        Kokkos::fence();
        return mat;
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        // This function returns the eigen sparse matrix representation of the gate.
        auto numRows = _matrix.numRows();
        auto numCols = _matrix.numCols();
        ComplexMatrix mat(numRows, numCols);
        Kokkos::View<Complex**,
                     Kokkos::LayoutRight,
                     Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            mat_view(reinterpret_cast<Complex*>(mat.data()), numRows, numCols);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(numRows, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                UINT i = team.league_rank();
                UINT start_idx = _matrix.graph.row_map(i);
                UINT end_idx = _matrix.graph.row_map(i + 1);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, start_idx, end_idx), [&](const UINT idx) {
                        auto j = _matrix.graph.entries(idx);
                        auto val = _matrix.values(idx);
                        mat_view(i, j) = Kokkos::complex<double>(val.real(), val.imag());
                    });
            });
        Kokkos::fence();
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        std::vector<UINT> target_index_list = this->get_target_qubit_list();
        std::vector<UINT> control_index_list, control_value_list;
        for (auto info : control_qubit_info_list) {
            control_index_list.push_back(info.index());
            control_value_list.push_back(info.control_value());
        }
        sparse_matrix_gate(
            target_index_list, control_index_list, control_value_list, _matrix, state_vector);
    }
};

class DenseMatrixGateImpl : public GateBase {
    Matrix _matrix;
    bool _is_unitary;
    std::vector<TargetQubitInfo> target_qubit_info_list;
    std::vector<ControlQubitInfo> control_qubit_info_list;

public:
    DenseMatrixGateImpl(ComplexMatrix matrix,
                        const std::vector<UINT>& target_qubit_index_list,
                        const std::vector<UINT>& control_qubit_index_list,
                        bool is_unitary = false)
        : GateBase(), _is_unitary(is_unitary) {
        _matrix = std::move(convert_external_matrix_to_internal_matrix(matrix));
        for (UINT i = 0; i < target_qubit_index_list.size(); i++) {
            target_qubit_info_list.push_back(TargetQubitInfo(target_qubit_index_list[i]));
        }
        for (UINT i = 0; i < control_qubit_index_list.size(); i++) {
            control_qubit_info_list.push_back(ControlQubitInfo(control_qubit_index_list[i], 1));
        }
    }

    std::vector<UINT> get_target_qubit_list() const override {
        std::vector<UINT> target_qubit_list;
        for (const auto& target : target_qubit_info_list) {
            target_qubit_list.push_back(target.index());
        }
        return target_qubit_list;
    }
    std::vector<UINT> get_control_qubit_list() const override {
        std::vector<UINT> control_qubit_list;
        for (const auto& control : control_qubit_info_list) {
            control_qubit_list.push_back(control.index());
        }
        return control_qubit_list;
    }

    void add_control_qubit(UINT control_qubit_index, UINT value) {
        control_qubit_info_list.push_back(ControlQubitInfo(control_qubit_index, value));
    }

    Gate get_inverse() const override {
        UINT rows = _matrix.extent(0);
        UINT cols = _matrix.extent(1);
        ComplexMatrix mat_eigen = convert_internal_matrix_to_external_matrix(_matrix);
        ComplexMatrix inv_eigen;
        if (_is_unitary) {
            inv_eigen = mat_eigen.adjoint();
        } else {
            inv_eigen = mat_eigen.lu().solve(ComplexMatrix::Identity(rows, cols));
        }
        return std::make_shared<const DenseMatrixGateImpl>(
            inv_eigen, get_target_qubit_list(), get_control_qubit_list(), _is_unitary);
    }
    std::optional<Matrix> get_matrix_internal() const { return _matrix; }

    std::optional<ComplexMatrix> get_matrix() const override {
        return convert_internal_matrix_to_external_matrix(_matrix);
    }

    void update_quantum_state(StateVector& state_vector) const override {
        std::vector<UINT> target_index_list = get_target_qubit_list();
        std::vector<UINT> control_index_list;
        std::vector<UINT> control_value_list;
        for (const auto& control : control_qubit_info_list) {
            control_index_list.push_back(control.index());
            control_value_list.push_back(control.control_value());
        }

        for (auto i : target_index_list) check_qubit_within_bounds(state_vector, i);
        for (auto i : control_index_list) check_qubit_within_bounds(state_vector, i);
        dense_matrix_gate(
            target_index_list, control_index_list, control_value_list, _matrix, state_vector);
    }
};
}  // namespace internal

using OneQubitMatrixGate = internal::GatePtr<internal::OneQubitMatrixGateImpl>;
using TwoQubitMatrixGate = internal::GatePtr<internal::TwoQubitMatrixGateImpl>;
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl>;
using DensityMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl>;
}  // namespace scaluq
