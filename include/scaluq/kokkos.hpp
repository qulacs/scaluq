#pragma once

namespace scaluq {
void initialize();
void finalize();
bool is_initialized();
bool is_finalized();

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_kokkos_hpp(nb::module_& m) {
    m.def("finalize",
          &finalize,
          "Terminate the Kokkos execution environment. Release the resources.\n\n.. note:: "
          "Finalization fails if there exists `StateVector` allocated. You must use "
          "`StateVector` only inside inner scopes than the usage of `finalize` or delete all of "
          "existing `StateVector`.\n\n.. note:: This is "
          "automatically called when the program exits. If you call this manually, you cannot use "
          "most of scaluq's functions until the program exits.");
    m.def("is_finalized", &is_initialized, "Return true if `finalize()` is already called.");
}
}  // namespace internal
#endif
}  // namespace scaluq
