#include <Kokkos_Core.hpp>
#include <scaluq/kokkos.hpp>

namespace scaluq {
void initialize() { Kokkos::initialize(); }
void finalize() { Kokkos::finalize(); }
bool is_initialized() { return Kokkos::is_initialized(); }
bool is_finalized() { return Kokkos::is_finalized(); }
void synchronize() { Kokkos::fence(); }

std::vector<ConcurrentStream> create_streams(const std::vector<double>& weights) {
    auto instances =
        Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), weights);
    std::vector<ConcurrentStream> out;
    out.reserve(instances.size());
    for (const auto& inst : instances) {
        out.emplace_back(inst);
    }
    return out;
}

void synchronize(const ConcurrentStream& stream) {
#if defined(SCALUQ_USE_CUDA)
    stream.fence("scaluq::synchronize");
#else
    Kokkos::fence("scaluq::synchronize");
#endif
}

void synchronize(const std::vector<ConcurrentStream>& streams) {
#if defined(SCALUQ_USE_CUDA)
    for (const auto& stream : streams) {
        stream.fence("scaluq::synchronize");
    }
#else
    Kokkos::fence("scaluq::synchronize");
#endif
}
}  // namespace scaluq
