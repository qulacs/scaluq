#pragma once

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

public:
    OneQubitMatrixGateImpl(UINT target, const std::array<std::array<Complex, 2>, 2>& matrix)
        : OneQubitGateBase(target) {
        _matrix.val[0][0] = matrix[0][0];
        _matrix.val[0][1] = matrix[0][1];
        _matrix.val[1][0] = matrix[1][0];
        _matrix.val[1][1] = matrix[1][1];
    }

    std::array<std::array<Complex, 2>, 2> matrix() {
        return {_matrix.val[0][0], _matrix.val[0][1], _matrix.val[1][0], _matrix.val[1][1]};
    }

    Gate copy() const override { return std::make_shared<OneQubitMatrixGateImpl>(*this); }
    Gate get_inverse() const override {
        return std::make_shared<OneQubitMatrixGateImpl>(
            _target,
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

    void update_quantum_state(StateVector& state_vector) const override;
};

class TwoQubitMatrixGateImpl : public TwoQubitGateBase {
    matrix_4_4 _matrix;

public:
    TwoQubitMatrixGateImpl(UINT target1,
                           UINT target2,
                           const std::array<std::array<Complex, 4>, 4>& matrix)
        : TwoQubitGateBase(target1, target2) {
        for (UINT i : std::views::iota(4)) {
            for (UINT j : std::views::iota(4)) {
                _matrix.val[i][j] = matrix[i][j];
            }
        }
    }

    std::array<std::array<Complex, 4>, 4> matrix() {
        std::array<std::array<Complex, 4>, 4> matrix;
        for (UINT i : std::views::iota(4)) {
            for (UINT j : std::views::iota(4)) {
                matrix[i][j] = _matrix.val[i][j];
            }
        }
        return matrix;
    }

    Gate copy() const override { return std::make_shared<TwoQubitMatrixGateImpl>(*this); }
    Gate get_inverse() const override {
        std::array<std::array<Complex, 4>, 4> matrix_dag;
        for (UINT i : std::views::iota(4)) {
            for (UINT j : std::views::iota(4)) {
                matrix_dag[i][j] = Kokkos::conj(_matrix.val[j][i]);
            }
        }
        return std::make_shared<TwoQubitMatrixGateImpl>(_target1, _target2, matrix_dag);
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

    void update_quantum_state(StateVector& state_vector) const override;
};

class CrsMatrixGateImpl : public GateBase {
    CrsMatrix _matrix;
    std::vector<TargetQubitInfo> target_qubit_index_list;
    std::vector<ControlQubitInfo> control_qubit_index_list;

public:
    CrsMatrixGateImpl() : GateBase() {}

    std::vector<UINT> get_target_qubit_list() const override {
        std::vector<UINT> target_qubit_list;
        for (const auto& target : target_qubit_index_list) {
            target_qubit_list.push_back(target.index());
        }
        return target_qubit_list;
    }
    std::vector<UINT> get_control_qubit_list() const override {
        std::vector<UINT> control_qubit_list;
        for (const auto& control : control_qubit_index_list) {
            control_qubit_list.push_back(control.index());
        }
        return control_qubit_list;
    }

    void add_controle_qubit(UINT control_qubit_index, UINT value) {
        control_qubit_index_list.push_back(ControlQubitInfo(control_qubit_index, value));
    }

    Gate copy() const override { return std::make_shared<CrsMatrixGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<CrsMatrixGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override { return std::nullopt; }

    void update_quantum_state(StateVector& state_vector) const override {
        Kokkos::View<scaluq::Complex*> state = state_vector._raw;
        Kokkos::View<scaluq::Complex*> buffer1("buffer1", state_vector.dim());
        Kokkos::View<scaluq::Complex*> buffer2("buffer2", state_vector.dim());

        const UINT target_qubit_index_count = target_qubit_index_list.size();
        const UINT matrix_dim = 1ULL << target_qubit_index_count;
        const std::vector<UINT> matrix_mask_list =
            create_matrix_mask_list(this->get_control_qubit_list(), target_qubit_index_count);
        const std::vector<UINT> sorted_insert_index_list =
            create_sorted_ui_list(this->get_control_qubit_list());
        const UINT loop_dim = state_vector.dim() >> target_qubit_index_count;

        for (UINT state_index = 0; state_index < loop_dim; ++state_index) {
            UINT basis_0 = state_index;
            // create base index
            for (UINT cursor = 0; cursor < target_qubit_index_count; ++cursor) {
                UINT insert_index = sorted_insert_index_list[cursor];
                basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index, insert_index);
            }

            // fetch vector
            for (UINT j = 0; j < matrix_dim; ++j) {
                buffer1[j] = state[basis_0 ^ matrix_mask_list[j]];
            }

            spmv(_matrix, buffer1, buffer2);

            for (UINT j = 0; j < matrix_dim; ++j) {
                state[basis_0 ^ matrix_mask_list[j]] = buffer2[j];
            }
        }
    }
};

class DenseMatrixGateImpl : public GateBase {
    DenseMatrix _matrix;
    std::vector<TargetQubitInfo> target_qubit_index_list;
    std::vector<ControlQubitInfo> control_qubit_index_list;

public:
    DenseMatrixGateImpl() : GateBase() {}

    std::vector<UINT> get_target_qubit_list() const override {
        std::vector<UINT> target_qubit_list;
        for (const auto& target : target_qubit_index_list) {
            target_qubit_list.push_back(target.index());
        }
        return target_qubit_list;
    }
    std::vector<UINT> get_control_qubit_list() const override {
        std::vector<UINT> control_qubit_list;
        for (const auto& control : control_qubit_index_list) {
            control_qubit_list.push_back(control.index());
        }
        return control_qubit_list;
    }

    void add_controle_qubit(UINT control_qubit_index, UINT value) {
        control_qubit_index_list.push_back(ControlQubitInfo(control_qubit_index, value));
    }

    Gate copy() const override { return std::make_shared<DenseMatrixGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<DenseMatrixGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override { return std::nullopt; }

    void update_quantum_state(StateVector& state_vector) const override {
        // std::vector<UINT> target_index;
        // for (const auto& target : target_qubit_index_list) {
        //     target_index.push_back(target.index());
        // }
        // std::vector<UINT> control_index;
        // std::vector<UINT> control_value;
        // for (const auto& control : control_qubit_index_list) {
        //     control_qubit_index_list.push_back(control.index());
        // }

        // if(this->target_qubit_index_list.size() == 1){
        //     // no control qubit
        //     if(this->control_qubit_index_list.size() == 0){
        //         single_qubit_dense_matrix_gate(target_index[0], _matrix, state_vector);
        //     }
        // }
    }
};
}  // namespace internal

using OneQubitMatrixGate = internal::GatePtr<internal::OneQubitMatrixGateImpl>;
using TwoQubitMatrixGate = internal::GatePtr<internal::TwoQubitMatrixGateImpl>;
using CrsMatrixGate = internal::GatePtr<internal::CrsMatrixGateImpl>;
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl>;
}  // namespace scaluq
