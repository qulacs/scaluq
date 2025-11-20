#pragma once

#include "types.hpp"

namespace scaluq {
void initialize();
void finalize();
bool is_initialized();
bool is_finalized();
void synchronize();
}  // namespace scaluq

#ifdef SCALUQ_USE_NANOBIND
#include "../python/docstring.hpp"
namespace scaluq::internal {
void bind_kokkos_hpp(nb::module_& m) {
    m.def(
        "finalize",
        &finalize,
        DocString()
            .desc("Terminate the Kokkos execution environment. Release the resources.")
            .note("Finalization fails if there exists `StateVector` allocated. You must use "
                  "`StateVector` only inside inner scopes than the usage of `finalize` or delete "
                  "all of existing `StateVector`.")
            .note("This is automatically called when the program exits. If you call this manually, "
                  "you cannot use most of scaluq's functions until the program exits.")
            .build_as_google_style()
            .c_str());
    m.def("is_finalized",
          &is_initialized,
          "Return true if :func:`~scaluq.finalize()` is already called.");
    m.def("synchronize",
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
