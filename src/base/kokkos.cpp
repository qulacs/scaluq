#include <Kokkos_Core.hpp>
#include <scaluq/kokkos.hpp>

namespace scaluq {
void initialize() { Kokkos::initialize(); }
void finalize() { Kokkos::finalize(); }
bool is_initialized() { return Kokkos::is_initialized(); }
bool is_finalized() { return Kokkos::is_finalized(); }
void synchronize() { Kokkos::fence(); }
void synchronize(const ConcurrentStream& stream) { stream.fence("scaluq::synchronize"); }

std::vector<ConcurrentStream> create_default_streams(const std::vector<double>& weights) {
    auto instances =
        Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), weights);
    std::vector<ConcurrentStream> out;
    out.reserve(instances.size());
    for (const auto& inst : instances) {
        out.emplace_back(inst);
    }
    return out;
}

std::vector<ConcurrentStream> create_host_streams(const std::vector<double>& weights) {
    auto instances =
        Kokkos::Experimental::partition_space(Kokkos::DefaultHostExecutionSpace(), weights);
    std::vector<ConcurrentStream> out;
    out.reserve(instances.size());
    for (const auto& inst : instances) {
        out.emplace_back(inst);
    }
    return out;
}

}  // namespace scaluq
