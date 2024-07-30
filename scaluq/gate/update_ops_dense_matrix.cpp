#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"

namespace scaluq {
namespace internal {
void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    const matrix_2_2& matrix,
                                    StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 1, KOKKOS_LAMBDA(const UINT it) {
            UINT basis_0 = internal::insert_zero_to_basis_index(it, target_qubit_index);
            UINT basis_1 = basis_0 | (1ULL << target_qubit_index);
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex res0 = matrix.val[0][0] * val0 + matrix.val[0][1] * val1;
            Complex res1 = matrix.val[1][0] * val0 + matrix.val[1][1] * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
    Kokkos::fence();
}

void double_qubit_dense_matrix_gate(UINT target0,
                                    UINT target1,
                                    const matrix_4_4& matrix,
                                    StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 2, KOKKOS_LAMBDA(const UINT it) {
            UINT basis_0 = internal::insert_two_zero_to_basis_index(it, target0, target1);
            UINT basis_1 = basis_0 | (1ULL << target0);
            UINT basis_2 = basis_0 | (1ULL << target1);
            UINT basis_3 = basis_1 | (1ULL << target1);
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex val2 = state._raw[basis_2];
            Complex val3 = state._raw[basis_3];
            Complex res0 = matrix.val[0][0] * val0 + matrix.val[0][1] * val1 +
                           matrix.val[0][2] * val2 + matrix.val[0][3] * val3;
            Complex res1 = matrix.val[1][0] * val0 + matrix.val[1][1] * val1 +
                           matrix.val[1][2] * val2 + matrix.val[1][3] * val3;
            Complex res2 = matrix.val[2][0] * val0 + matrix.val[2][1] * val1 +
                           matrix.val[2][2] * val2 + matrix.val[2][3] * val3;
            Complex res3 = matrix.val[3][0] * val0 + matrix.val[3][1] * val1 +
                           matrix.val[3][2] * val2 + matrix.val[3][3] * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}

void single_qubit_dense_matrix_gate_view(UINT target_qubit_index,
                                         const Matrix& matrix,
                                         StateVector& state) {
    const UINT loop_dim = state.dim() >> 1;
    const UINT mask = (1ULL << target_qubit_index);
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
            UINT basis_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            UINT basis_1 = basis_0 + mask;
            Complex v0 = state._raw[basis_0];
            Complex v1 = state._raw[basis_1];
            state._raw[basis_0] = matrix(0, 0) * v0 + matrix(0, 1) * v1;
            state._raw[basis_1] = matrix(1, 0) * v0 + matrix(1, 1) * v1;
        });
    Kokkos::fence();
}

void double_qubit_dense_matrix_gate(UINT target_qubit_index1,
                                    UINT target_qubit_index2,
                                    const Matrix& matrix,
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

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
            UINT basis_index_0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                 ((state_index & high_mask) << 2);
            UINT basis_index_1 = basis_index_0 + target_mask1;
            UINT basis_index_2 = basis_index_0 + target_mask2;
            UINT basis_index_3 = basis_index_1 + target_mask2;

            Complex cval0 = state._raw[basis_index_0];
            Complex cval1 = state._raw[basis_index_1];
            Complex cval2 = state._raw[basis_index_2];
            Complex cval3 = state._raw[basis_index_3];

            state._raw[basis_index_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1 +
                                        matrix(0, 2) * cval2 + matrix(0, 3) * cval3;
            state._raw[basis_index_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1 +
                                        matrix(1, 2) * cval2 + matrix(1, 3) * cval3;
            state._raw[basis_index_2] = matrix(2, 0) * cval0 + matrix(2, 1) * cval1 +
                                        matrix(2, 2) * cval2 + matrix(2, 3) * cval3;
            state._raw[basis_index_3] = matrix(3, 0) * cval0 + matrix(3, 1) * cval1 +
                                        matrix(3, 2) * cval2 + matrix(3, 3) * cval3;
        });
    Kokkos::fence();
}

void single_qubit_control_single_qubit_dense_matrix_gate(UINT control_qubit_index,
                                                         UINT control_value,
                                                         UINT target_qubit_index,
                                                         const Matrix& matrix,
                                                         StateVector& state) {
    const UINT loop_dim = state.dim() >> 2;
    const UINT target_mask = 1ULL << target_qubit_index;
    const UINT control_mask = 1ULL << control_qubit_index;

    const auto [min_qubit_index, max_qubit_index] =
        std::minmax(control_qubit_index, target_qubit_index);
    const UINT min_qubit_mask = 1ULL << min_qubit_index;
    const UINT max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const UINT low_mask = min_qubit_mask - 1;
    const UINT mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const UINT high_mask = ~(max_qubit_mask - 1);

    if (target_qubit_index == 0) {
        Kokkos::parallel_for(
            loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
                UINT basis_index = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                   ((state_index & high_mask) << 2) + control_mask * control_value;

                Complex cval0 = state._raw[basis_index];
                Complex cval1 = state._raw[basis_index + 1];

                state._raw[basis_index] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state._raw[basis_index + 1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
            });
        Kokkos::fence();
    } else if (control_qubit_index == 0) {
        Kokkos::parallel_for(
            loop_dim, KOKKOS_LAMBDA(const UINT state_index) {
                UINT basis_index_0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                     ((state_index & high_mask) << 2) +
                                     control_mask * control_value;
                UINT basis_index_1 = basis_index_0 + target_mask;

                Complex cval0 = state._raw[basis_index_0];
                Complex cval1 = state._raw[basis_index_1];

                state._raw[basis_index_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state._raw[basis_index_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
            });
        Kokkos::fence();
    } else {
        Kokkos::parallel_for(
            loop_dim << 1, KOKKOS_LAMBDA(UINT state_index) {
                state_index *= 2;
                UINT basis_index_0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                                     ((state_index & high_mask) << 2) +
                                     control_mask * control_value;
                UINT basis_index_1 = basis_index_0 + target_mask;

                Complex cval0 = state._raw[basis_index_0];
                Complex cval1 = state._raw[basis_index_1];
                Complex cval2 = state._raw[basis_index_0 + 1];
                Complex cval3 = state._raw[basis_index_1 + 1];

                state._raw[basis_index_0] = matrix(0, 0) * cval0 + matrix(0, 1) * cval1;
                state._raw[basis_index_1] = matrix(1, 0) * cval0 + matrix(1, 1) * cval1;
                state._raw[basis_index_0 + 1] = matrix(0, 0) * cval2 + matrix(0, 1) * cval3;
                state._raw[basis_index_1 + 1] = matrix(1, 0) * cval2 + matrix(1, 1) * cval3;
            });
        Kokkos::fence();
    }
}

void single_qubit_control_multi_qubit_dense_matrix_gate(
    UINT control_qubit_index,
    UINT control_value,
    const std::vector<UINT>& target_qubit_index_list,
    const Matrix& matrix,
    StateVector& state) {
    const UINT target_qubit_index_count = target_qubit_index_list.size();
    const UINT matrix_dim = 1ULL << target_qubit_index_count;
    const UINT insert_index_count = target_qubit_index_count + 1;
    const UINT control_mask = (1ULL << control_qubit_index) * control_value;
    const UINT loop_dim = state.dim() >> insert_index_count;

    std::vector<UINT> matrix_mask_list = create_matrix_mask_list(target_qubit_index_list);
    std::vector<UINT> sorted_insert_index_list =
        create_sorted_ui_list_value(target_qubit_index_list, control_qubit_index);
    Kokkos::View<UINT*> matrix_mask_view = convert_host_vector_to_device_view(matrix_mask_list);
    Kokkos::View<UINT*> sorted_insert_index_view =
        convert_host_vector_to_device_view(sorted_insert_index_list);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            UINT basis_0 = team.league_rank();
            for (UINT i = 0; i < insert_index_count; ++i) {
                UINT insert_index = sorted_insert_index_view(i);
                basis_0 = insert_zero_to_basis_index(basis_0, insert_index);
            }
            basis_0 ^= control_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](const UINT y) {
                buffer[y] = 0;
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](const UINT x, Complex& inner_sum) {
                        inner_sum += matrix(y, x) * state._raw[basis_0 ^ matrix_mask_view(x)];
                    },
                    sum);
                buffer[y] = sum;
            });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [=](const UINT y) {
                state._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
            });
        });
    Kokkos::fence();
}

void multi_qubit_control_single_qubit_dense_matrix_gate(
    const std::vector<UINT>& control_qubit_index_list,
    const std::vector<UINT>& control_value_list,
    UINT target_qubit_index,
    const Matrix& matrix,
    StateVector& state_vector) {
    UINT control_qubit_index_count = control_qubit_index_list.size();
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
                state_index <<= 1;
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
    Kokkos::fence();
}

void multi_qubit_control_multi_qubit_dense_matrix_gate(
    const std::vector<UINT>& control_qubit_index_list,
    const std::vector<UINT>& control_value_list,
    const std::vector<UINT>& target_qubit_index_list,
    const Matrix& matrix,
    StateVector& state) {
    const UINT control_qubit_index_count = control_qubit_index_list.size();
    const UINT target_qubit_index_count = target_qubit_index_list.size();
    const UINT matrix_dim = 1ULL << target_qubit_index_count;
    const UINT insert_index_count = target_qubit_index_count + control_qubit_index_count;
    UINT control_mask = create_control_mask(control_qubit_index_list, control_value_list);
    const UINT loop_dim = state.dim() >> insert_index_count;
    std::vector<UINT> matrix_mask_list = create_matrix_mask_list(target_qubit_index_list);
    std::vector<UINT> sorted_insert_index_list =
        create_sorted_ui_list_list(target_qubit_index_list,
                                   target_qubit_index_count,
                                   control_qubit_index_list,
                                   control_qubit_index_count);
    Kokkos::View<UINT*> matrix_mask_view = convert_host_vector_to_device_view(matrix_mask_list);
    Kokkos::View<UINT*> sorted_insert_index_view =
        convert_host_vector_to_device_view(sorted_insert_index_list);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            UINT basis_0 = team.league_rank();
            for (UINT i = 0; i < insert_index_count; ++i) {
                UINT insert_index = sorted_insert_index_view(i);
                basis_0 = insert_zero_to_basis_index(basis_0, insert_index);
            }
            basis_0 ^= control_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](UINT y) {
                buffer[y] = 0;
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](UINT x, Complex& inner_sum) {
                        inner_sum += matrix(y, x) * state._raw[basis_0 ^ matrix_mask_view(x)];
                    },
                    sum);
                buffer[y] = sum;
            });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [=](const UINT y) {
                state._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
            });
        });
    Kokkos::fence();
}

void multi_qubit_dense_matrix_gate_parallel(const std::vector<UINT>& target_qubit_index_list,
                                            const Matrix& matrix,
                                            StateVector& state) {
    std::vector<UINT> sort_array, mask_array;
    create_shift_mask_list_from_list_buf(target_qubit_index_list, sort_array, mask_array);

    const UINT target_qubit_index_count = target_qubit_index_list.size();
    const UINT matrix_dim = 1ULL << target_qubit_index_count;
    const std::vector<UINT> matrix_mask_list = create_matrix_mask_list(target_qubit_index_list);

    const UINT loop_dim = state.dim() >> target_qubit_index_count;
    auto mask_view = convert_host_vector_to_device_view(mask_array);
    auto matrix_mask_view = convert_host_vector_to_device_view(matrix_mask_list);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            auto buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            UINT basis_0 = team.league_rank();
            for (UINT cursor = 0; cursor < target_qubit_index_count; ++cursor) {
                basis_0 = (basis_0 & mask_view[cursor]) + ((basis_0 & (~mask_view[cursor])) << 1);
            }

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](UINT y) {
                buffer[y] = 0;
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](UINT x, Complex& inner_sum) {
                        inner_sum += matrix(y, x) * state._raw[basis_0 ^ matrix_mask_view(x)];
                    },
                    sum);
                buffer[y] = sum;
            });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [=](const UINT y) {
                state._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
            });
        });
    Kokkos::fence();
}

void multi_qubit_dense_matrix_gate(const std::vector<UINT>& target_qubit_index_list,
                                   const Matrix& matrix,
                                   StateVector& state_vector) {
    UINT target_qubit_index_count = target_qubit_index_list.size();
    if (target_qubit_index_count == 1) {
        single_qubit_dense_matrix_gate_view(target_qubit_index_list[0], matrix, state_vector);
        return;
    } else if (target_qubit_index_count == 2) {
        double_qubit_dense_matrix_gate(
            target_qubit_index_list[0], target_qubit_index_list[1], matrix, state_vector);
        return;
    } else {
        multi_qubit_dense_matrix_gate_parallel(target_qubit_index_list, matrix, state_vector);
        return;
    }
}

void dense_matrix_gate(std::vector<UINT> target_index_list,
                       std::vector<UINT> control_index_list,
                       std::vector<UINT> control_value_list,
                       const Matrix& matrix,
                       StateVector& state) {
    if (target_index_list.size() == 1) {
        if (control_index_list.size() == 0) {
            single_qubit_dense_matrix_gate_view(target_index_list[0], matrix, state);
        } else if (control_index_list.size() == 1) {
            single_qubit_control_single_qubit_dense_matrix_gate(
                control_index_list[0], control_value_list[0], target_index_list[0], matrix, state);
        } else {
            multi_qubit_control_single_qubit_dense_matrix_gate(
                control_index_list, control_value_list, target_index_list[0], matrix, state);
        }
    } else {
        if (control_index_list.size() == 0) {
            multi_qubit_dense_matrix_gate(target_index_list, matrix, state);
        } else if (control_index_list.size() == 1) {
            single_qubit_control_multi_qubit_dense_matrix_gate(
                control_index_list[0], control_value_list[0], target_index_list, matrix, state);
        } else {
            // multiple control qubit
            multi_qubit_control_multi_qubit_dense_matrix_gate(
                control_index_list, control_value_list, target_index_list, matrix, state);
        }
    }
}
}  // namespace internal
}  // namespace scaluq
