#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <state/state_vector.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qulacs;

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T>
struct type_caster<Kokkos::complex<T>> {
    NB_TYPE_CASTER(Kokkos::complex<T>, const_name("complex"))

    template <bool Recursive = true>
    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        (void)flags;
        (void)cleanup;

        if (PyComplex_Check(src.ptr())) {
            value = Kokkos::complex<T>((T)PyComplex_RealAsDouble(src.ptr()),
                                       (T)PyComplex_ImagAsDouble(src.ptr()));
            return true;
        }

        if constexpr (Recursive) {
            if (!PyFloat_CheckExact(src.ptr()) && !PyLong_CheckExact(src.ptr()) &&
                PyObject_HasAttrString(src.ptr(), "imag")) {
                try {
                    object tmp = handle(&PyComplex_Type)(src);
                    return from_python<false>(tmp, flags, cleanup);
                } catch (...) {
                    return false;
                }
            }
        }

        make_caster<T> caster;
        if (caster.from_python(src, flags, cleanup)) {
            value = Kokkos::complex<T>(caster.operator cast_t<T>());
            return true;
        }

        return true;
    }

    template <typename T2>
    static handle from_cpp(T2 &&value, rv_policy policy, cleanup_list *cleanup) noexcept {
        (void)policy;
        (void)cleanup;

        return PyComplex_FromDoubles((double)value.real(), (double)value.imag());
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

NB_MODULE(qulacs_core, m) {
    nb::class_<InitializationSettings>(m, "InitializationSettings")
        .def(nb::init<>())
        .def("set_num_threads", &InitializationSettings::set_num_threads)
        .def("has_num_threads", &InitializationSettings::has_num_threads)
        .def("get_num_threads", &InitializationSettings::get_num_threads)
        .def("set_device_id", &InitializationSettings::set_device_id)
        .def("has_device_id", &InitializationSettings::has_device_id)
        .def("get_device_id", &InitializationSettings::get_device_id)
        .def("set_map_device_id_by", &InitializationSettings::set_map_device_id_by)
        .def("has_map_device_id_by", &InitializationSettings::has_map_device_id_by)
        .def("get_map_device_id_by", &InitializationSettings::get_map_device_id_by)
        .def("set_disable_warnings", &InitializationSettings::set_disable_warnings)
        .def("has_disable_warnings", &InitializationSettings::has_disable_warnings)
        .def("get_disable_warnings", &InitializationSettings::get_disable_warnings)
        .def("set_print_configuration", &InitializationSettings::set_print_configuration)
        .def("has_print_configuration", &InitializationSettings::has_print_configuration)
        .def("get_print_configuration", &InitializationSettings::get_print_configuration)
        .def("set_tune_internals", &InitializationSettings::set_tune_internals)
        .def("has_tune_internals", &InitializationSettings::has_tune_internals)
        .def("get_tune_internals", &InitializationSettings::get_tune_internals)
        .def("set_tools_help", &InitializationSettings::set_tools_help)
        .def("has_tools_help", &InitializationSettings::has_tools_help)
        .def("get_tools_help", &InitializationSettings::get_tools_help)
        .def("set_tools_libs", &InitializationSettings::set_tools_libs)
        .def("has_tools_libs", &InitializationSettings::has_tools_libs)
        .def("get_tools_libs", &InitializationSettings::get_tools_libs)
        .def("set_tools_args", &InitializationSettings::set_tools_args)
        .def("has_tools_args", &InitializationSettings::has_tools_args)
        .def("get_tools_args", &InitializationSettings::get_tools_args);

    m.def("initialize", &qulacs::initialize, "settings"_a = InitializationSettings());
    m.def("finalize", &qulacs::finalize);

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
