#pragma once

#include <ranges>
#include <vector>

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
    std::vector<UINT> target_qubit_index_list, control_qubit_index_list;

public:
    CrsMatrixGateImpl() : GateBase() {}

    std::vector<UINT> get_target_qubit_list() const override { return target_qubit_index_list; }
    std::vector<UINT> get_control_qubit_list() const override { return control_qubit_index_list; }

    void add_controle_qubit(UINT control_qubit_index) {
        control_qubit_index_list.push_back(control_qubit_index);
    }

    Gate copy() const override { return std::make_shared<CrsMatrixGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<CrsMatrixGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override { return std::nullopt; }

    void update_quantum_state(StateVector& state_vector) const override {}

    // void update_quantum_state(StateVector& state_vector) const override {
    //     Kokkos::View<scaluq::Complex*> state = state_vector._raw;
    //     Kokkos::View<scaluq::Complex*> buffer1("buffer1", state_vector.dim());
    //     Kokkos::View<scaluq::Complex*> buffer2("buffer2", state_vector.dim());

    //     const UINT target_qubit_index_count = target_qubit_index_list.size();
    //     const UINT matrix_dim = 1ULL << target_qubit_index_count;
    //     const std::vector<UINT> matrix_mask_list =
    //         create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    //     const std::vector<UINT> sorted_insert_index_list =
    //         create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);
    //     const UINT loop_dim = state_vector.dim() >> target_qubit_index_count;

    //     Kokkos::parallel_for(
    //         loop_dim, KOKKOS_LAMBDA(const UINT& state_index) {
    //             UINT basis_0 = state_index;
    //             // create base index
    //             Kokkos::parallel_for(
    //                 target_qubit_index_count, KOKKOS_LAMBDA(const UINT& cursor) {
    //                     UINT insert_index = sorted_insert_index_list[cursor];
    //                     basis_0 =
    //                         insert_zero_to_basis_index(basis_0, 1ULL << insert_index,
    //                         insert_index);
    //                 });

    //             // fetch vector
    //             Kokkos::parallel_for(
    //                 matrix_dim, KOKKOS_LAMBDA(const UINT& y) {
    //                     buffer1[y] = state[basis_0 ^ matrix_mask_list[y]];
    //                 });

    //             spmv(_matrix, buffer1, buffer2);
    //             buffer1 = buffer2;

    //             Kokkos::parallel_for(
    //                 matrix_dim, KOKKOS_LAMBDA(const UINT& y) {
    //                     state[basis_0 ^ matrix_mask_list[y]] = buffer1[y];
    //                 });
    //         });
    // }
};

// class DenseMatrixGateImpl : public GateBase {
//     DenseMatrix _matrix;
//     std::vector<UINT> target_qubit_index_list, control_qubit_index_list;

// public:
//     DenseMatrixGateImpl() : GateBase() {}

//     std::vector<UINT> get_target_qubit_list() const override { return target_qubit_index_list; }
//     std::vector<UINT> get_control_qubit_list() const override { return control_qubit_index_list;
//     }

//     void add_controle_qubit(UINT control_qubit_index) {
//         control_qubit_index_list.push_back(control_qubit_index);
//     }

//     Gate copy() const override { return std::make_shared<CrsMatrixGateImpl>(*this); }
//     Gate get_inverse() const override { return std::make_shared<CrsMatrixGateImpl>(*this); }
//     std::optional<ComplexMatrix> get_matrix() const override { return std::nullopt; }

//     void update_quantum_state(StateVector& state_vector) const override {}
// };
}  // namespace internal

using OneQubitMatrixGate = internal::GatePtr<internal::OneQubitMatrixGateImpl>;
using TwoQubitMatrixGate = internal::GatePtr<internal::TwoQubitMatrixGateImpl>;
}  // namespace scaluq
