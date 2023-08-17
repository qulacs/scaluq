#include <cpusim/state_vector_cpu.hpp>

#include "internal/update_ops.hpp"

#ifdef OPENMP
#include "internal/omp_manager.hpp"
#endif

void x_gate(UINT target_qubit_index, StateVectorCpu& state) {
#ifdef OPENMP
    OmpManager::get_instance().set_num_threads(state.dim(), 13);
#endif

    const int mask = (1ULL << target_qubit_index);
    const int mask_low = mask - 1;
    const int mask_high = ~mask_low;
    if (target_qubit_index == 0) {
        int basis_index = 0;
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (basis_index = 0; basis_index < state.dim(); basis_index += 2) {
            Complex temp = state[basis_index];
            state[basis_index] = state[basis_index + 1];
            state[basis_index + 1] = temp;
        }
    } else {
        int state_index = 0;
        const int loop_dim = state.dim() / 2;
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            const int basis_index_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            const int basis_index_1 = basis_index_0 + mask;
            const Complex temp0 = state[basis_index_0];
            const Complex temp1 = state[basis_index_0 + 1];

            state[basis_index_0] = state[basis_index_1];
            state[basis_index_0 + 1] = state[basis_index_1 + 1];
            state[basis_index_1] = temp0;
            state[basis_index_1 + 1] = temp1;
        }
    }

#ifdef OPENMP
    OmpManager::get_instance().reset_num_threads();
#endif
}
