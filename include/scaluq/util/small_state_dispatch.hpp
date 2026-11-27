#pragma once

#include <Kokkos_Core.hpp>

#include "../types.hpp"

namespace scaluq::internal {

template <typename State, typename Policy, typename Functor>
void parallel_for_small_state(const State& state,
                              const char* label,
                              const Policy& policy,
                              const Functor& functor) {
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (State::space == ExecutionSpace::Host) {
        const auto limit = get_small_state_serial_threshold();
        if (limit != 0 && state.flat_dim() <= limit) {
            for (auto i = policy.begin(); i < policy.end(); ++i) {
                functor(i);
            }
            return;
        }
    }
#else
    (void)state;
#endif
    Kokkos::parallel_for(label, policy, functor);
}
}  // namespace scaluq::internal
