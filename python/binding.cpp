#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "docstring.hpp"

namespace nb = nanobind;
using namespace nb::literals;

#define SCALUQ_USE_NANOBIND

#include <scaluq/all.hpp>

using namespace scaluq;
using namespace std::string_literals;

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T>
struct type_caster<Kokkos::complex<T>> {
    NB_TYPE_CASTER(Kokkos::complex<T>, const_name("complex"))

    template <bool Recursive = true>
    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
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

        return false;
    }

    template <typename T2>
    static handle from_cpp(T2&& value, rv_policy policy, cleanup_list* cleanup) noexcept {
        (void)policy;
        (void)cleanup;

        return PyComplex_FromDoubles((double)value.real(), (double)value.imag());
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

void cleanup() {
    if (!is_finalized()) finalize();
}

template <std::floating_point Fp>
void bind_on_precision(nb::module_& m, const char* submodule_name) {
    auto mp = m.def_submodule(
        submodule_name,
        (std::ostringstream("module for ") << submodule_name << "precision").str().c_str());

    internal::bind_state_state_vector_hpp<Fp>(mp);
    internal::bind_state_state_vector_batched_hpp<Fp>(mp);

    auto mgate = mp.def_submodule("gate", "Define gates.");

    internal::bind_gate_gate_hpp<Fp>(mp);
    internal::bind_gate_gate_standard_hpp<Fp>(mp);
    internal::bind_gate_gate_matrix_hpp<Fp>(mp);
    internal::bind_gate_gate_pauli_hpp<Fp>(mp);
    internal::bind_gate_gate_factory_hpp<Fp>(mgate);
    internal::bind_gate_param_gate_hpp<Fp>(mp);
    internal::bind_gate_param_gate_standard_hpp<Fp>(mp);
    internal::bind_gate_param_gate_pauli_hpp<Fp>(mp);
    internal::bind_gate_param_gate_probablistic_hpp<Fp>(mp);
    internal::bind_gate_param_gate_factory<Fp>(mgate);
    // internal::bind_gate_merge_gate_hpp<Fp>(mp);

    internal::bind_circuit_circuit_hpp<Fp>(mp);

    internal::bind_operator_pauli_operator_hpp<Fp>(mp);
    internal::bind_operator_operator_hpp<Fp>(mp);
}

NB_MODULE(scaluq_core, m) {
    internal::bind_kokkos_hpp(m);
    internal::bind_gate_gate_hpp_without_precision(m);

    bind_on_precision<double>(m, "f64");
    bind_on_precision<float>(m, "f32");

    initialize();
    std::atexit(&cleanup);
}
