#pragma once

#include <Kokkos_Core.hpp>
#include <mutex>

#include "../types.hpp"

namespace scaluq::internal {
// Serial's implementation has process-wide scratch storage. Serialize these launches.
std::mutex& small_state_serial_mutex();

template <typename State, typename Policy, typename Functor>
void parallel_for_small_state(const State& state,
                              const char* label,
                              const Policy& policy,
                              const Functor& functor) {
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (State::space == ExecutionSpace::Host) {
        const auto limit = get_small_state_serial_threshold();
        if (limit != 0 && state.flat_dim() <= limit) {
            std::lock_guard<std::mutex> lock(small_state_serial_mutex());
            Kokkos::parallel_for(
                label, Kokkos::RangePolicy<Kokkos::Serial>(policy.begin(), policy.end()), functor);
            return;
        }
    }
#else
    (void)state;
#endif
    Kokkos::parallel_for(label, policy, functor);
}
}  // namespace scaluq::internal
