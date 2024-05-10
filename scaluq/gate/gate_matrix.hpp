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
    std::vector<TargetQubitInfo> target_qubit_info_list;
    std::vector<ControlQubitInfo> control_qubit_info_list;

public:
    CrsMatrixGateImpl() : GateBase() {}

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

    void add_controle_qubit(UINT control_qubit_index, UINT value) {
        control_qubit_info_list.push_back(ControlQubitInfo(control_qubit_index, value));
    }

    Gate copy() const override { return std::make_shared<CrsMatrixGateImpl>(*this); }
    Gate get_inverse() const override {
        // Not implemented
        return nullopt;
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        // TODO: Implement
        return std::nullopt;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        Kokkos::View<scaluq::Complex*> state = state_vector._raw;
        Kokkos::View<scaluq::Complex*> buffer1("buffer1", state_vector.dim());
        Kokkos::View<scaluq::Complex*> buffer2("buffer2", state_vector.dim());

        const UINT target_qubit_index_count = target_qubit_info_list.size();
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
    std::vector<TargetQubitInfo> target_qubit_info_list;
    std::vector<ControlQubitInfo> control_qubit_info_list;

public:
    DenseMatrixGateImpl() : GateBase() {}

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

    void add_controle_qubit(UINT control_qubit_index, UINT value) {
        control_qubit_info_list.push_back(ControlQubitInfo(control_qubit_index, value));
    }

    Gate copy() const override { return std::make_shared<DenseMatrixGateImpl>(*this); }
    Gate get_inverse() const override {
        // TODO: implement
        return nullopt; 
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        // TODO: implement
        return std::nullopt;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        std::vector<UINT> target_index_list = get_target_qubit_list();
        std::vector<UINT> control_index_list;
        std::vector<UINT> control_value_list;
        for (const auto& control : control_qubit_info_list) {
            control_index_list.push_back(control.index());
            control_value_list.push_back(control.value());
        }

        if (this->target_qubit_index_list.size() == 1) {
            // no control qubit
            if (this->control_qubit_info_list.size() == 0) {
                single_qubit_dense_matrix_gate(target_index_list[0], _matrix, state_vector);
            } else if (this->control_qubit_info_list.size() == 1) {
                single_qubit_control_single_qubit_dense_matrix_gate(control_index_list[0],
                                                                    control_value_list[0],
                                                                    target_index_list[0],
                                                                    _matrix,
                                                                    state_vector);
            } else {
                multi_qubit_control_single_qubit_dense_matrix_gate(control_index_list,
                                                                   control_value_list,
                                                                   target_index_list[0],
                                                                   _matrix,
                                                                   state_vector);
            }
        } else {
            // multi qubit dense matrix gate
            // no control qubit
            if (this->control_qubit_info_list.size() == 0) {
                multi_qubit_dense_matrix_gate(target_index_list, _matrix, state_vector);
            } else if (this->control_qubit_info_list.size() == 1) {
                single_qubit_control_multi_qubit_dense_matrix_gate(control_index_list[0],
                                                                   control_value_list[0],
                                                                   target_index_list,
                                                                   _matrix,
                                                                   state_vector);
            } else {
                // multiple control qubit
                multi_qubit_control_multi_qubit_dense_matrix_gate(control_index_list,
                                                                  control_value_list,
                                                                  target_index_list,
                                                                  _matrix,
                                                                  state_vector);
            }
        }
    }
};

void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    DenseMatrix& matrix,
                                    StateVector& state_vector) {
    check_qubit_within_bounds(state_vector, target_qubit_index);
    const UINT loop_dim = state_vector.dim() >> 1;
    const UINT mask = (1ULL << target_qubit_index);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;

    Kokkos::View<scaluq::Complex*> state = state_vector._raw;
    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
            UINT basis_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            UINT basis_1 = basis_0 + mask;
            Complex v0 = state[basis_0];
            Complex v1 = state[basis_1];
            state[basis_0] = matrix(0, 0) * v0 + matrix(0, 1) * v1;
            state[basis_1] = matrix(1, 0) * v0 + matrix(1, 1) * v1;
        });
}

void double_qubit_dense_matrix_gate(UINT target_qubit_index1,
                                    UINT target_qubit_index2,
                                    DenseMatrix& matrix,
                                    StateVector& state) {
    const auto [min_qubit_index, max_qubit_index] =
        std::minmax(target_qubit_index1, target_qubit_index2);
    const UINT min_qubit_mask = 1ULL << min_qubit_index;
    const UINT max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const UINT low_mask = min_qubit_mask - 1;
    const UINT mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const UINT high_mask = ~(max_qubit_mask - 1);

    const UINT target_mask1 = 1ULL << target_qubit_index1;
    const UINT target_mask2 = 1ULL << target_qubit_index2;

    const UINT loop_dim = state.dim() >> 2;
    Kokkos::View<scaluq::Complex*> state = state._raw;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
            UINT basis_index_0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                 ((state_index & high_mask) << 2);
            UINT basis_index_1 = basis_index_0 + target_mask1;
            UINT basis_index_2 = basis_index_0 + target_mask2;
            UINT basis_index_3 = basis_index_0 + target_mask1 + target_mask2;

            Complex cval0 = state[basis_index_0];
            Complex cval1 = state[basis_index_1];
            Complex cval2 = state[basis_index_2];
            Complex cval3 = state[basis_index_3];

            state[basis_index_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1 +
                                   matrix(0, 2) * cval2 + matrix(0, 3) * cval3;
            state[basis_index_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1 +
                                   matrix(1, 2) * cval2 + matrix(1, 3) * cval3;
            state[basis_index_2] = matrix(2, 0) * cval0 + matrix(2, 1) * cval1 +
                                   matrix(2, 2) * cval2 + matrix(2, 3) * cval3;
            state[basis_index_3] = matrix(3, 0) * cval0 + matrix(3, 1) * cval1 +
                                   matrix(3, 2) * cval2 + matrix(3, 3) * cval3;
        });
}

void single_qubit_control_single_qubit_dense_matrix_gate(UINT control_qubit_index,
                                                         UINT control_value,
                                                         UINT target_qubit_index,
                                                         DenseMatrix& matrix,
                                                         StateVector& state_vector) {
    check_qubit_within_bounds(state_vector, control_qubit_index);
    const UINT loop_dim = state_vector.dim() >> 2;
    const UINT target_mask = 1ULL << target_qubit_index;
    const UINT control_mask = 1ULL << control_qubit_index;

    const auto [min_qubit_index, max_qubit_index] =
        std::minmax(control_qubit_index, target_qubit_index);
    const UINT min_qubit_mask = 1ULL << min_qubit_index;
    const UINT max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const UINT low_mask = min_qubit_mask - 1;
    const UINT mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const UINT high_mask = ~(max_qubit_mask - 1);
    Kokkos::View<scaluq::Complex*> state = state_vector._raw;

    if (target_qubit_index == 0) {
        Kokkos::parallel_for(
            loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
                UINT basis_index = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                   ((state_index & high_mask) << 2) + control_mask * control_value;

                Complex cval0 = state[basis_index];
                Complex cval1 = state[basis_index + 1];

                state[basis_index] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state[basis_index + 1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
            });
    } else if (control_qubit_index == 0) {
        Kokkos::parallel_for(
            loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
                UINT basis_index_0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                     ((state_index & high_mask) << 2) +
                                     control_mask * control_value;
                UINT basis_index_1 = basis_index_0 + target_mask;

                Complex cval0 = state[basis_index_0];
                Complex cval1 = state[basis_index_1];

                state[basis_index_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state[basis_index_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
            });
    } else {
        Kokkos::parallel_for(
            loop_dim << 1, KOKKOS_LAMBDA(UINT state_index) {
                state_index *= 2;
                UINT basis_index_0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                     ((state_index & high_mask) << 2) +
                                     control_mask * control_value;
                UINT basis_index_1 = basis_index_0 + target_mask;

                Complex cval0 = state[basis_index_0];
                Complex cval1 = state[basis_index_1];
                Complex cval2 = state[basis_index_0 + 1];
                Complex cval3 = state[basis_index_1 + 1];

                state[basis_index_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state[basis_index_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
                state[basis_index_0 + 1] = matrix(0, 0) * cval2 + matrix(0, 1) * cval3;
                state[basis_index_1 + 1] = matrix(1, 0) * cval2 + matrix(1, 1) * cval3;
            });
    }
}

void single_qubit_control_multi_qubit_dense_matrix_gate(UINT control_qubit_index,
                                                        UINT control_value,
                                                        std::vector<UINT> target_qubit_index_list,
                                                        DenseMatrix matrix,
                                                        StateVector& state) {
    const UINT target_qubit_index_count = target_qubit_index_list.size();
    const UINT matrix_dim = 1ULL << target_qubit_index_count;
    std::vector<UINT> matrix_mask_list =
        create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);

    const UINT insert_index_count = target_qubit_index_count + 1;
    std::vector<UINT> sorted_insert_index_list = create_sorted_ui_list_value(
        target_qubit_index_list, target_qubit_index_count, control_qubit_index);
    const UINT control_mask = (1ULL << control_qubit_index) * control_value;
    const UINT loop_dim = state.dim() >> insert_index_count;
    Kokkos::View<scaluq::Complex*> state = state._raw;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
            UINT basis_0 = state_index;
            for (UINT i = 0; i < insert_index_count; ++i) {
                UINT insert_index = sorted_insert_index_list[i];
                basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index, insert_index);
            }

            basis_0 ^= control_mask;

            for (UINT j = 0; j < matrix_dim; ++j) {
                buffer[j] = 0;
                for (UINT k = 0; k < matrix_dim; ++k) {
                    buffer[j] += matrix(k, j) * state[basis_0 ^ matrix_mask_list[k]];
                }
            }

            for (UINT j = 0; j < matrix_dim; ++j) {
                state[basis_0 ^ matrix_mask_list[j]] = buffer[j];
            }
        });
}

void multi_qubit_control_single_qubit_dense_matrix_gate(
    const std::vector<UINT>& control_qubit_index_list,
    const std::vector<UINT>& control_value_list,
    UINT target_qubit_index,
    DenseMatrix& matrix,
    StateVector& state_vector) {
    if (control_qubit_index_count == 1) {
        single_qubit_control_single_qubit_dense_matrix_gate(control_qubit_index_list[0],
                                                            control_value_list[0],
                                                            target_qubit_index,
                                                            matrix,
                                                            state_vector);
        return;
    }

    std::vector<UINT> sort_array, mask_array;
    create_shift_mask_list_from_list_and_value_buf(
        control_qubit_index_list, target_qubit_index, sort_array, mask_array);
    const UINT target_mask = 1ULL << target_qubit_index;
    const UINT control_mask = create_control_mask(control_qubit_index_list, control_value_list);

    const UINT insert_index_list_count = control_qubit_index_list.size() + 1;
    const UINT loop_dim = state_vector.dim() >> insert_index_list_count;
    Kokkos::View<scaluq::Complex*> state = state_vector._raw;
    if (target_qubit_index == 0) {
        Kokkos::parallel_for(
            loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
                UINT basis_0 = state_index;
                for (UINT i = 0; i < insert_index_list_count; ++i) {
                    basis_0 = (basis_0 & mask_array[i]) + ((basis_0 & (~mask_array[i])) << 1);
                }
                basis_0 += control_mask;

                Complex cval0 = state[basis_0];
                Complex cval1 = state[basis_0 + 1];

                state[basis_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state[basis_0 + 1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
            });
    } else if (sort_array[0] == 0) {
        Kokkos::parallel_for(
            loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
                UINT basis_0 = state_index;
                for (UINT i = 0; i < insert_index_list_count; ++i) {
                    basis_0 = (basis_0 & mask_array[i]) + ((basis_0 & (~mask_array[i])) << 1);
                }
                basis_0 += control_mask;
                UINT basis_1 = basis_0 + target_mask;

                Complex cval0 = state[basis_0];
                Complex cval1 = state[basis_1];

                state[basis_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state[basis_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
            });
    } else {
        Kokkos::parallel_for(
            loop_dim << 1, KOKKOS_LAMBDA(UINT state_index) {
                state_index *= 2;
                UINT basis_0 = state_index;
                for (UINT i = 0; i < insert_index_list_count; ++i) {
                    basis_0 = (basis_0 & mask_array[i]) + ((basis_0 & (~mask_array[i])) << 1);
                }
                basis_0 += control_mask;
                UINT basis_1 = basis_0 + target_mask;

                Complex cval0 = state[basis_0];
                Complex cval1 = state[basis_1];
                Complex cval2 = state[basis_0 + 1];
                Complex cval3 = state[basis_1 + 1];

                state[basis_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state[basis_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
                state[basis_0 + 1] = matrix(0, 0) * cval2 + matrix(0, 1) * cval3;
                state[basis_1 + 1] = matrix(1, 0) * cval2 + matrix(1, 1) * cval3;
            });
    }
}

void multi_qubit_control_multi_qubit_dense_matrix_gate(
    const std::vector<UINT> control_qubit_index_list,
    const std::vector<UINT> control_value_list,
    const std::vector<UINT> target_qubit_index_list,
    DenseMatrix& matrix,
    StateVector& state) {
    const UINT control_qubit_index_count = control_qubit_index_list.size();
    const UINT target_qubit_index_count = target_qubit_index_list.size();
    const UINT matrix_dim = 1ULL << target_qubit_index_count;
    std::vector<UINT> matrix_mask_list =
        create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);

    const UINT insert_index_count = target_qubit_index_count + control_qubit_index_count;
    std::vector<UINT> sorted_insert_index_list =
        create_sorted_ui_list_list(target_qubit_index_list,
                                   target_qubit_index_count,
                                   control_qubit_index_list,
                                   control_qubit_index_count);
    UINT control_mask = create_control_mask(control_qubit_index_list, control_value_list);
    const UINT loop_dim = state.dim() >> insert_index_count;
    Kokkos::View<scaluq::Complex*> state = state._raw;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
            UINT basis_0 = state_index;
            for (UINT i = 0; i < insert_index_count; ++i) {
                UINT insert_index = sorted_insert_index_list[i];
                basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index, insert_index);
            }
            basis_0 ^= control_mask;

            for (UINT j = 0; j < matrix_dim; ++j) {
                buffer[j] = 0;
                for (UINT k = 0; k < matrix_dim; ++k) {
                    buffer[j] += matrix(k, j) * state[basis_0 ^ matrix_mask_list[k]];
                }
            }

            for (UINT j = 0; j < matrix_dim; ++j) {
                state[basis_0 ^ matrix_mask_list[j]] = buffer[j];
            }
        });
}

void multi_qubit_dense_matrix_gate(const std::vector<UINT> target_qubit_index_list,
                                   DenseMatrix& matrix,
                                   StateVector& state_vector) {
    UINT target_qubit_index_count = target_qubit_index_list.size();
    if (target_qubit_index_count == 1) {
        single_qubit_dense_matrix_gate(target_qubit_index_list[0], matrix, state_vector);
        return;
    } else if (target_qubit_index_count == 2) {
        double_qubit_dense_matrix_gate(
            target_qubit_index_list[0], target_qubit_index_list[1], matrix, state_vector);
        return;
    } else {
        std::vector<UINT> sort_array, mask_array;
        create_shift_mask_list_from_list_buf(target_qubit_index_list, sort_array, mask_array);
        const UINT matrix_dim = 1ULL << target_qubit_index_count;
        const UINT loop_dim = state_vector.dim() >> target_qubit_index_count;
        const std::vector<UINT> matrix_mask_list =
            create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
        Kokkos::View<scaluq::Complex*> state = state_vector._raw;
        Kokkos::View<scaluq::Complex*> buffer =
            Kokkos::View<scaluq::Complex*>("buffer", matrix_dim);
        Kokkos::parallel_for(
            loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
                UINT basis_0 = state_index;
                for (UINT i = 0; i < target_qubit_index_count; ++i) {
                    basis_0 = (basis_0 & mask_array[i]) + ((basis_0 & (~mask_array[i])) << 1);
                }

                for (UINT j = 0; j < matrix_dim; ++j) {
                    buffer[j] = 0;
                    for (UINT k = 0; k < matrix_dim; ++k) {
                        buffer[j] += matrix(k, j) * state[basis_0 ^ matrix_mask_list[k]];
                    }
                }

                for (UINT j = 0; j < matrix_dim; ++j) {
                    state[basis_0 ^ matrix_mask_list[j]] = buffer[j];
                }
            });
    }
}

}  // namespace internal

using OneQubitMatrixGate = internal::GatePtr<internal::OneQubitMatrixGateImpl>;
using TwoQubitMatrixGate = internal::GatePtr<internal::TwoQubitMatrixGateImpl>;
using CrsMatrixGate = internal::GatePtr<internal::CrsMatrixGateImpl>;
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl>;
}  // namespace scaluq
