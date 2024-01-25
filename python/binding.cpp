#include <nanobind/nanobind.h>

#include <state/state_vector.hpp>

namespace nb = nanobind;
using namespace qulacs;

NB_MODULE(qulacs_core, m) {
    nb::class_<StateVector>(m, "StateVector")
        .def(nb::init<>())
        .def(nb::init<UINT>())
        .def(nb::init<const StateVector &>())
        .def("set_zero_state", &StateVector::set_zero_state)
        .def("set_zero_norm_state", &StateVector::set_zero_norm_state)
        .def("set_computational_basis", &StateVector::set_computational_basis)
        .def("amplitudes", &StateVector::amplitudes)
        .def("n_qubits", &StateVector::n_qubits)
        .def("dim", &StateVector::dim)
        .def("compute_squared_norm", &StateVector::compute_squared_norm)
        .def("normalize", &StateVector::normalize)
        .def("get_zero_probability", &StateVector::get_zero_probability)
        .def("get_marginal_probability", &StateVector::get_marginal_probability)
        .def("get_entropy", &StateVector::get_entropy)
        .def("add_state_vector", &StateVector::add_state_vector)
        .def("add_state_vector_with_coef", &StateVector::add_state_vector_with_coef)
        .def("multiply_coef", &StateVector::multiply_coef)
        .def("sampling", &StateVector::sampling)
        .def("to_string", &StateVector::to_string)
        .def("load", &StateVector::load)
        .def("__getitem__", [](const StateVector &s, int index) { return s[index]; })
        .def("__setitem__",
             [](StateVector &s, int index, const Complex &value) { s[index] = value; })
        .def("__str__", &StateVector::to_string);

    m.def("Haar_random_state",
          static_cast<StateVector (*)(UINT, UINT)>(&StateVector::Haar_random_state));
    m.def("Haar_random_state", static_cast<StateVector (*)(UINT)>(&StateVector::Haar_random_state));
}
