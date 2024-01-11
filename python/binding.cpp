#include <nanobind/nanobind.h>

#include <state/state_vector.hpp>

namespace nb = nanobind;
using namespace qulacs;

NB_MODULE(qulacs_core, m) { nb::class_<StateVector>(m, "StateVector"); }
