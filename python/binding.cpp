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

#include <stdfloat>

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
struct type_caster<Complex<T>> {
    NB_TYPE_CASTER(Complex<T>, const_name("complex"))

    template <bool Recursive = true>
    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        (void)flags;
        (void)cleanup;

        if (PyComplex_Check(src.ptr())) {
            value = Complex<T>((T)PyComplex_RealAsDouble(src.ptr()),
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
            value = Complex<T>(caster.operator cast_t<T>());
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

// template <>
// struct dtype_traits<scaluq::F16> {
//     static constexpr dlpack::dtype value{
//         (uint8_t)dlpack::dtype_code::Float,  // type code
//         16,                                  // size in bits
//         1                                    // lanes (simd), usually set to 1
//     };
//     static constexpr auto name = const_name("float16");
// };

// template <>
// struct dtype_traits<scaluq::BF16> {
//     static constexpr dlpack::dtype value{
//         (uint8_t)dlpack::dtype_code::Float,  // type code
//         16,                                  // size in bits
//         1                                    // lanes (simd), usually set to 1
//     };
//     static constexpr auto name = const_name("bfloat16");
// };

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
    internal::bind_gate_param_gate_hpp_without_precision(m);

#ifdef SCALUQ_FLOAT16
    bind_on_precision<F16>(m, "f16");
#endif
#ifdef SCALUQ_FLOAT32
    bind_on_precision<F32>(m, "f32");
#endif
#ifdef SCALUQ_FLOAT64
    bind_on_precision<F64>(m, "f64");
#endif
#ifdef SCALUQ_BFLOAT16
    bind_on_precision<BF16>(m, "bf16");
#endif

    m.def(
        "precision_available",
        [](std::string_view precision) {
            if (precision == "f16") {
#ifdef SCALUQ_FLOAT16
                return true;
#else
                return false;
#endif
            }
            if (precision == "f32") {
#ifdef SCALUQ_FLOAT32
                return true;
#else
                return false;
#endif
            }
            if (precision == "f64") {
#ifdef SCALUQ_FLOAT64
                return true;
#else
                return false;
#endif
            }
            if (precision == "bf16") {
#ifdef SCALUQ_BFLOAT16
                return true;
#else
                return false;
#endif
            }
            throw std::runtime_error("precision_available: Unknown precision name.");
        },
        DocString()
            .desc("Return the precision is supported.")
            .arg("precision",
                 "str",
                 "precision name",
                 "This must be one of `f16` `f32` `f64` `bf16`.")
            .ret("bool", "the precision is supported")
            .ex(DocString::Code{">>> precision_available('f64')",
                                "True",
                                ">>> precision_available('bf16')",
                                "False"})
            .build_as_google_style()
            .c_str());

    initialize();
    std::atexit(&cleanup);
}
