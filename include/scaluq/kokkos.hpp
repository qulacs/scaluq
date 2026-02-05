#pragma once

#include <Kokkos_Core.hpp>
#include <vector>

#include "types.hpp"

namespace scaluq {
void initialize();
void finalize();
bool is_initialized();
bool is_finalized();
void synchronize();

inline std::vector<ConcurrentStream> create_streams(const std::vector<double>& weights) {
    auto instances =
        Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), weights);
    std::vector<ConcurrentStream> out;
    out.reserve(instances.size());
    for (const auto& inst : instances) {
        out.emplace_back(inst);
    }
    return out;
}
}  // namespace scaluq

#ifdef SCALUQ_USE_NANOBIND
#include "../python/docstring.hpp"
#include "../types.hpp"
namespace scaluq::internal {
void bind_kokkos_hpp(nb::module_& m) {
    nb::class_<ConcurrentStream>(
        m,
        "ConcurrentStream",
        DocString()
            .desc("Execution space instance for concurrent stream control.")
            .build_as_google_style()
            .c_str())
        .def("fence",
             &ConcurrentStream::fence,
             "name"_a = "scaluq::ConcurrentStream::fence",
             DocString()
                 .desc("Fence the execution space instance.")
                 .arg("name", "str", "Fence label")
                 .build_as_google_style()
                 .c_str());
    m.def("create_streams",
          &create_streams,
          "weights"_a,
          DocString()
              .desc("Create concurrent streams by partitioning the default execution space.")
              .arg("weights", "list[float]", "Partition weights")
              .ret("list[ConcurrentStream]", "Concurrent stream instances")
              .build_as_google_style()
              .c_str());
    m.def("initialize",
          &initialize,
          DocString()
              .desc("Initialize the Kokkos execution environment.")
              .note("This is automatically called when the program starts. You do not have to call "
                    "this manually unless you have called `finalize` before.")
              .build_as_google_style()
              .c_str())
        .def(
            "finalize",
            &finalize,
            DocString()
                .desc("Terminate the Kokkos execution environment. Release the resources.")
                .note(
                    "Finalization fails if there exists `StateVector` allocated. You must use "
                    "`StateVector` only inside inner scopes than the usage of `finalize` or delete "
                    "all of existing `StateVector`.")
                .note("This is automatically called when the program exits. If you call this "
                      "manually, "
                      "you cannot use most of scaluq's functions until the program exits.")
                .build_as_google_style()
                .c_str())
        .def("is_initialized",
             &is_initialized,
             "Return true if :func:`~scaluq.initialize()` is already called.")
        .def("is_finalized",
             &is_finalized,
             "Return true if :func:`~scaluq.finalize()` is already called.")
        .def("synchronize",
             &synchronize,
             DocString()
                 .desc("Synchronize the device if the execution space is not host.")
                 .note("This function is required to ensure that all operations on device are "
                       "finished when you measure the elapsed time of some operations on device.")
                 .build_as_google_style()
                 .c_str());
}
}  // namespace scaluq::internal
#endif
