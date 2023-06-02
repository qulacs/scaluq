#include <pybind11/pybind11.h>

#include "state_vector.hpp"

PYBIND11_MODULE(qulacs_core, m)
{
    pybind11::class_<StateVector>(m, "StateVector");
}
