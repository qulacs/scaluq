#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
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

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

void cleanup() {
    if (!is_finalized()) finalize();
}

template <Precision Prec>
void bind_on_precision(nb::module_& m, const char* submodule_name) {
    std::ostringstream oss;
    oss << "module for " << submodule_name << "precision";
    auto mp = m.def_submodule(submodule_name, oss.str().c_str());

    internal::bind_state_state_vector_hpp<Prec>(mp);
    internal::bind_state_state_vector_batched_hpp<Prec>(mp);

    auto mgate = mp.def_submodule("gate", "Define gates.");

    internal::bind_gate_gate_hpp<Prec>(mp);
    internal::bind_gate_gate_standard_hpp<Prec>(mp);
    internal::bind_gate_gate_matrix_hpp<Prec>(mp);
    internal::bind_gate_gate_pauli_hpp<Prec>(mp);
    internal::bind_gate_gate_factory_hpp<Prec>(mgate);
    internal::bind_gate_param_gate_hpp<Prec>(mp);
    internal::bind_gate_param_gate_standard_hpp<Prec>(mp);
    internal::bind_gate_param_gate_pauli_hpp<Prec>(mp);
    internal::bind_gate_param_gate_probablistic_hpp<Prec>(mp);
    internal::bind_gate_param_gate_factory<Prec>(mgate);
    internal::bind_gate_merge_gate_hpp<Prec>(mp);

    internal::bind_circuit_circuit_hpp<Prec>(mp);

    internal::bind_operator_pauli_operator_hpp<Prec>(mp);
    internal::bind_operator_operator_hpp<Prec>(mp);
}

NB_MODULE(scaluq_core, m) {
    internal::bind_kokkos_hpp(m);
    internal::bind_gate_gate_hpp_without_precision(m);
    internal::bind_gate_param_gate_hpp_without_precision(m);

#ifdef SCALUQ_FLOAT16
    bind_on_precision<Precision::F16>(m, "f16");
#endif
#ifdef SCALUQ_FLOAT32
    bind_on_precision<Precision::F32>(m, "f32");
#endif
#ifdef SCALUQ_FLOAT64
    bind_on_precision<Precision::F64>(m, "f64");
#endif
#ifdef SCALUQ_BFLOAT16
    bind_on_precision<Precision::BF16>(m, "bf16");
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
