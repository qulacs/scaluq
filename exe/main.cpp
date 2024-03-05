#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../qulacs/all.hpp"
#include "../qulacs/util/utility.hpp"
#include "../tests/gate/util.hpp"

using namespace qulacs;
using namespace std;

const auto eps = 1e-12;
using CComplex = std::complex<double>;

KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index, UINT insert_index) {
    UINT mask = (1ULL << insert_index) - 1;
    UINT temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
}

void pauli_rotation_func(PauliOperator* pauli, double angle, StateVector& state) {
    auto [bit_flip_mask_vector, phase_flip_mask_vector] = pauli->get_XZ_mask_representation();
    UINT bit_flip_mask = bit_flip_mask_vector.data_raw()[0];
    UINT phase_flip_mask = phase_flip_mask_vector.data_raw()[0];
    UINT global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    const double cosval = cos(-angle / 2);
    const double sinval = sin(-angle / 2);
    const Complex coef = pauli->get_coef();
    const auto& amplitudes = state.amplitudes_raw();
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            state.dim(), KOKKOS_LAMBDA(const UINT& state_idx) {
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    amplitudes[state_idx] *= cosval - Complex(0, 1) * sinval;
                } else {
                    amplitudes[state_idx] *= cosval + Complex(0, 1) * sinval;
                }
                amplitudes[state_idx] *= coef;
            });
        return;
    } else {
        const UINT mask = 1 << bit_flip_mask_vector.msb();
        const UINT mask_low = mask - 1;
        const UINT mask_high = ~mask_low;
        Kokkos::parallel_for(
            state.dim() / 2, KOKKOS_LAMBDA(const UINT& state_idx) {
                UINT basis_0 = (state_idx & mask_low) + ((state_idx & mask_high) << 1);
                UINT basis_1 = basis_0 ^ bit_flip_mask;
                std::cout << "i: " << state_idx << ", basis_0: " << basis_0
                          << ", basis_1: " << basis_1 << std::endl;

                int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) % 2;
                int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) % 2;

                // fetch values
                Complex cval_0 = amplitudes[basis_0];
                Complex cval_1 = amplitudes[basis_1];

                // set values
                amplitudes[basis_0] =
                    cosval * cval_0 +
                    Complex(0, 1) * sinval * cval_1 *
                        (PHASE_M90ROT()).val[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                amplitudes[basis_1] =
                    cosval * cval_1 +
                    Complex(0, 1) * sinval * cval_0 *
                        (PHASE_M90ROT()).val[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
                amplitudes[basis_0] *= coef;
                amplitudes[basis_1] *= coef;
            });
    }
}

void test_idx(PauliOperator* pauli, StateVector& state) {
    auto [bit_flip_mask_vector, phase_flip_mask_vector] = pauli->get_XZ_mask_representation();
    UINT bit_flip_mask = bit_flip_mask_vector.data_raw()[0];
    UINT phase_flip_mask = phase_flip_mask_vector.data_raw()[0];
    const UINT mask = 1 << bit_flip_mask_vector.msb();
    const UINT insert_idx = bit_flip_mask_vector.msb();
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;
    vector<UINT> basis_0_list, basis_0_list_inserted;
    for (UINT state_idx = 0; state_idx < state.dim() / 2; state_idx++) {
        UINT basis_0 = (state_idx & mask_low) + ((state_idx & mask_high) << 1);
        UINT basis_0_inserted = insert_zero_to_basis_index(state_idx, insert_idx);
        basis_0_list.push_back(basis_0);
        basis_0_list_inserted.push_back(basis_0_inserted);
    }
    for (int i = 0; i < basis_0_list.size(); i++) {
        cout << "basis_0_list[" << i << "] = " << basis_0_list[i] << ", basis_0_list_inserted[" << i
             << "] = " << basis_0_list_inserted[i]
             << ", diff: " << basis_0_list[i] - basis_0_list_inserted[i] << endl;
    }
    cout << "test_idx passed" << endl;
}

void run() {
    UINT n_qubits = 10;
    auto state = StateVector::Haar_random_state(n_qubits);
    vector<UINT> pauli_ids, targets;
    for (UINT i = 0; i < n_qubits; i++) {
        pauli_ids.push_back(i % 4);
        targets.push_back(i);
    }
    auto pauli = PauliOperator(targets, pauli_ids, 1.0);
    test_idx(&pauli, state);
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
