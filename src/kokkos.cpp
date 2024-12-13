#include <Kokkos_Core.hpp>
#include <scaluq/kokkos.hpp>

namespace scaluq {
void initialize() { Kokkos::initialize(); }
void finalize() { Kokkos::finalize(); }
bool is_initialized() { return Kokkos::is_initialized(); }
bool is_finalized() { return Kokkos::is_finalized(); }
}  // namespace scaluq
