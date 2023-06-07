#include <pybind11/pybind11.h>

#include <cpusim/state_vector_cpu.hpp>

PYBIND11_MODULE(qulacs_core, m) { pybind11::class_<StateVectorCpu>(m, "StateVectorCpu"); }
