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

template <Precision Prec, ExecutionSpace Space>
void bind_on_precision_and_space(nb::module_& mspace, const char* submodule_name) {
    std::ostringstream oss;
    oss << "module for " << submodule_name << "precision";
    auto mp = mspace.def_submodule(submodule_name, oss.str().c_str());

    internal::bind_state_state_vector_hpp<Prec, Space>(mp);
    internal::bind_state_state_vector_batched_hpp<Prec, Space>(mp);

    auto mgate = mp.def_submodule("gate", "Define gates.");

    auto gate_base_def = internal::bind_gate_gate_hpp<Prec, Space>(mp);
    internal::bind_gate_gate_standard_hpp<Prec, Space>(mp, gate_base_def);
    internal::bind_gate_gate_matrix_hpp<Prec, Space>(mp, gate_base_def);
    internal::bind_gate_gate_pauli_hpp<Prec, Space>(mp, gate_base_def);
    internal::bind_gate_gate_factory_hpp<Prec, Space>(mgate);
    auto param_gate_base_def = internal::bind_gate_param_gate_hpp<Prec, Space>(mp);
    internal::bind_gate_param_gate_standard_hpp<Prec, Space>(mp, param_gate_base_def);
    internal::bind_gate_param_gate_pauli_hpp<Prec, Space>(mp, param_gate_base_def);
    internal::bind_gate_param_gate_probabilistic_hpp<Prec, Space>(mp, param_gate_base_def);
    internal::bind_gate_param_gate_factory<Prec, Space>(mgate);
    internal::bind_gate_merge_gate_hpp<Prec, Space>(mp);

    internal::bind_circuit_circuit_hpp<Prec, Space>(mp);

    internal::bind_operator_pauli_operator_hpp<Prec, Space>(mp);
    internal::bind_operator_operator_hpp<Prec, Space>(mp);
}

NB_MODULE(scaluq_core, m) {
    internal::bind_kokkos_hpp(m);
    internal::bind_gate_gate_hpp_without_precision_and_space(m);
    internal::bind_gate_param_gate_hpp_without_precision_and_space(m);

    auto mdefault = m.def_submodule("default", "module for default execution space");
#ifdef SCALUQ_FLOAT16
    bind_on_precision_and_space<Precision::F16, ExecutionSpace::Default>(mdefault, "f16");
#endif
#ifdef SCALUQ_FLOAT32
    bind_on_precision_and_space<Precision::F32, ExecutionSpace::Default>(mdefault, "f32");
#endif
#ifdef SCALUQ_FLOAT64
    bind_on_precision_and_space<Precision::F64, ExecutionSpace::Default>(mdefault, "f64");
#endif
#ifdef SCALUQ_BFLOAT16
    bind_on_precision_and_space<Precision::BF16, ExecutionSpace::Default>(mdefault, "bf16");
#endif

#ifdef SCALUQ_USE_CUDA
    auto mhost = m.def_submodule("host", "module for host execution space");
#ifdef SCALUQ_FLOAT16
    bind_on_precision_and_space<Precision::F16, ExecutionSpace::Host>(mhost, "f16");
#endif
#ifdef SCALUQ_FLOAT32
    bind_on_precision_and_space<Precision::F32, ExecutionSpace::Host>(mhost, "f32");
#endif
#ifdef SCALUQ_FLOAT64
    bind_on_precision_and_space<Precision::F64, ExecutionSpace::Host>(mhost, "f64");
#endif
#ifdef SCALUQ_BFLOAT16
    bind_on_precision_and_space<Precision::BF16, ExecutionSpace::Host>(mhost, "bf16");
#endif
#endif

    m.def(
        "get_default_execution_space",
        []() -> std::string {
#ifdef SCALUQ_USE_CUDA
            return "cuda";
#else
            return "host";
#endif
        },
        DocString()
            .desc("Get the default execution space.")
            .ret("str", "the default execution space, `cuda` or `host`")
            .ex(DocString::Code{">>> get_default_execution_space()", "'cuda'"})
            .build_as_google_style()
            .c_str());

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
