#pragma once

#include <cstdint>

namespace scaluq {
void initialize();
void finalize();
bool is_initialized();
bool is_finalized();
void synchronize();
// Maximum number of amplitudes in a host state eligible for serial gate kernels.
// Zero disables serial dispatch (the default).
void set_small_state_serial_threshold(std::uint64_t max_amplitudes);
std::uint64_t get_small_state_serial_threshold();
}  // namespace scaluq

#ifdef SCALUQ_USE_NANOBIND
#include "../python/docstring.hpp"
namespace scaluq::internal {
void bind_kokkos_hpp(nb::module_& m) {
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
        .def("set_small_state_serial_threshold",
             &set_small_state_serial_threshold,
             nb::arg("max_amplitudes"),
             "Set the maximum host-state amplitude count for selected serial gate kernels; zero "
             "disables dispatch.")
        .def("get_small_state_serial_threshold",
             &get_small_state_serial_threshold,
             "Get the maximum host-state amplitude count for serial gate kernels.")
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
