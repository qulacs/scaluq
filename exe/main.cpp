#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

using namespace scaluq;
using namespace std;

// KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index, UINT insert_index) {
//     UINT mask = (1ULL << insert_index) - 1;
//     UINT temp_basis = (basis_index >> insert_index) << (insert_index + 1);
//     return temp_basis | (basis_index & mask);
// }

// KOKKOS_INLINE_FUNCTION UINT insert_zero_at_mask_positions(UINT basis_index, UINT insert_mask) {
//     for (UINT bit_mask = insert_mask; bit_mask;
//          bit_mask &= (bit_mask - 1)) {  // loop through set bits
//         UINT lower_mask = ~bit_mask & (bit_mask - 1);
//         UINT upper_mask = ~lower_mask;
//         basis_index = ((basis_index & upper_mask) << 1) | (basis_index & lower_mask);
//     }
//     return basis_index;
// }

// void x_gate(UINT target_qubit_index, StateVector& state) {
//     Kokkos::parallel_for(
//         state.dim() >> 1, KOKKOS_LAMBDA(UINT it) {
//             UINT i = insert_zero_to_basis_index(it, target_qubit_index);
//             Kokkos::Experimental::swap(state._raw[i], state._raw[i | (1ULL <<
//             target_qubit_index)]);
//         });
//     Kokkos::fence();
// }

// void cx_gate(UINT target_qubit_index, UINT control_qubit_index, StateVector& state) {
//     Kokkos::parallel_for(
//         state.dim() >> 2, KOKKOS_LAMBDA(UINT it) {
//             UINT i =
//                 internal::insert_zero_to_basis_index(it, target_qubit_index,
//                 control_qubit_index);
//             i |= 1ULL << control_qubit_index;
//             Kokkos::Experimental::swap(state._raw[i], state._raw[i | (1ULL <<
//             target_qubit_index)]);
//         });
//     Kokkos::fence();
// }

// void x_gate_control(UINT target_mask, UINT control_mask, StateVector& state) {
//     Kokkos::parallel_for(
//         state.dim() >> (1 + std::popcount(control_mask)), KOKKOS_LAMBDA(UINT it) {
//             UINT i = insert_zero_at_mask_positions(it, control_mask | target_mask) |
//             control_mask; Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
//         });
//     Kokkos::fence();
// }

// void run() {
//     const UINT n_qubits = 25, loop = 50;

//     std::vector<UINT> target, control;
//     Random rd;
//     for (UINT i = 0; i < loop; ++i) {
//         target.push_back(rd.int32() % n_qubits);
//         control.push_back(rd.int32() % n_qubits);
//     }

//     {  // warm up
//         auto state = StateVector::Haar_random_state(n_qubits);
//         for (UINT i = 0; i < loop; ++i) {
//             x_gate(target[i], state);
//         }
//     }

//     {
//         auto state1 = StateVector::Haar_random_state(n_qubits);
//         Kokkos::Timer tm;
//         for (UINT i = 0; i < loop; ++i) {
//             x_gate(target[i], state1);
//         }
//         std::cout << "x_gate            : " << tm.seconds() << std::endl;

//         auto state2 = StateVector::Haar_random_state(n_qubits);
//         tm.reset();
//         for (UINT i = 0; i < loop; ++i) {
//             x_gate_control(1ULL << target[i], 0, state2);
//         }
//         std::cout << "controlable x_gate: " << tm.seconds() << std::endl;

//         assert(state1.amplitudes() == state2.amplitudes());
//     }

//     {
//         auto state1 = StateVector::Haar_random_state(n_qubits);
//         Kokkos::Timer tm;
//         for (UINT i = 0; i < loop; ++i) {
//             cx_gate(target[i], control[i], state1);
//         }
//         std::cout << "cx_gate           : " << tm.seconds() << std::endl;

//         auto state2 = StateVector::Haar_random_state(n_qubits);
//         tm.reset();
//         for (UINT i = 0; i < loop; ++i) {
//             x_gate_control(1ULL << target[i], 1ULL << control[i], state2);
//         }
//         std::cout << "controlable x_gate: " << tm.seconds() << std::endl;

//         assert(state1.amplitudes() == state2.amplitudes());
//     }
// }

int main() {
    Kokkos::initialize();
    // run();
    Kokkos::finalize();
}
