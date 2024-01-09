#include <nanobind/nanobind.h>

#include <state/state_vector.hpp>

NB_MODULE(qulacs_core, m) { nb::class_<StateVector>(m, "StateVector"); }
