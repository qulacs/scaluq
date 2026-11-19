#include <Kokkos_Core.hpp>
#include <atomic>
#include <mutex>
#include <scaluq/kokkos.hpp>

namespace scaluq {
namespace {
std::atomic<std::uint64_t> small_state_threshold{0};
}
namespace internal {
std::mutex& small_state_serial_mutex() {
    static std::mutex mutex;
    return mutex;
}
}  // namespace internal
void set_small_state_serial_threshold(std::uint64_t max_amplitudes) {
    small_state_threshold.store(max_amplitudes, std::memory_order_relaxed);
}
std::uint64_t get_small_state_serial_threshold() {
    return small_state_threshold.load(std::memory_order_relaxed);
}
void initialize() { Kokkos::initialize(); }
void finalize() { Kokkos::finalize(); }
bool is_initialized() { return Kokkos::is_initialized(); }
bool is_finalized() { return Kokkos::is_finalized(); }
void synchronize() { Kokkos::fence(); }
}  // namespace scaluq
