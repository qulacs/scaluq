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

#include <all.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace scaluq;
using namespace std::string_literals;

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

        return false;
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

NB_MODULE(scaluq_core, m) {
    nb::class_<InitializationSettings>(
        m,
        "InitializationSettings",
        "Wrapper class of Kokkos's InitializationSettings.\\n.. note:: See details: "
        "https://kokkos.org/kokkos-core-wiki/API/core/initialize_finalize/"
        "InitializationSettings.html")
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

    m.def("initialize",
          &initialize,
          "settings"_a = InitializationSettings(),
          "**You must call this before any scaluq function.** Initialize the Kokkos execution "
          "environment.");
    m.def("finalize",
          &finalize,
          "Terminate the Kokkos execution environment. Release the resources.");
    m.def("is_initialized", &is_initialized, "Return true if `initialize()` is already called.");
    m.def("is_finalized", &is_initialized, "Return true if `finalize()` is already called.");

    nb::class_<StateVector>(m,
                            "StateVector",
                            "Vector representation of quantum state.\\n.. note:: Qubit index is "
                            "start from 0. If the amplitudes of $\\\\ket{b_{n-1}\\\\dots b_0}$ is "
                            "$b_i$, the state is $\\\\sum_i b_i 2^i$.")
        .def(nb::init<UINT>(),
             "Construct state vector with specified qubits, initialized with computational "
             "basis $\\ket{0\\dots0}$.")
        .def(nb::init<const StateVector &>(), "Constructing state vector by copying other state.")
        .def_static(
            "Haar_random_state",
            [](UINT n_qubits, std::optional<UINT> seed) {
                return StateVector::Haar_random_state(n_qubits,
                                                      seed.value_or(std::random_device{}()));
            },
            "n_qubits"_a,
            "seed"_a = std::nullopt,
            "Constructing state vector with Haar random state. If seed is not specified, the value "
            "from random device is used.")
        .def("set_amplitude_at_index",
             &StateVector::set_amplitude_at_index,
             "Manually set amplitude at one index.")
        .def(
            "get_amplitude_at_index",
            &StateVector::get_amplitude_at_index,
            "Get amplitude at one index.\\n.. note:: If you want to get all amplitudes, you should "
            "use `StateVector::amplitudes()`.")
        .def("set_zero_state",
             &StateVector::set_zero_state,
             "Initialize with computational basis $\\\\ket{00\\\\dots0}$.")
        .def("set_zero_norm_state",
             &StateVector::set_zero_norm_state,
             "Initialize with 0 (null vector).")
        .def("set_computational_basis",
             &StateVector::set_computational_basis,
             "Initialize with computational basis \\\\ket{\\\\mathrm{basis}}.")
        .def("amplitudes", &StateVector::amplitudes, "Get all amplitudes with as `list[complex]`.")
        .def("n_qubits", &StateVector::n_qubits, "Get num of qubits.")
        .def("dim",
             &StateVector::dim,
             "Get dimension of the vector ($=2^\\\\mathrm{n\\\\_qubits}$).")
        .def("get_squared_norm",
             &StateVector::get_squared_norm,
             "Get squared norm of the state. $\\\\braket{\\\\psi|\\\\psi}$.")
        .def("normalize",
             &StateVector::normalize,
             "Normalize state (let $\\\\braket{\\\\psi|\\\\psi} = 1$ by multiplying coef).")
        .def("get_zero_probability",
             &StateVector::get_zero_probability,
             "Get the probability to observe $\\\\ket{0}$ at specified index.")
        .def("get_marginal_probability",
             &StateVector::get_marginal_probability,
             "Get the marginal probability to observe as specified. Specify the result as n-length "
             "list. `0` and `1` represent the qubit is observed and get the value. `2` represents "
             "the qubit is not observed.")
        .def("get_entropy", &StateVector::get_entropy, "Get the entropy of the vector.")
        .def("add_state_vector",
             &StateVector::add_state_vector,
             "Add other state vector and make superposition. $\\\\ket{\\\\mathrm{this}} "
             "\\\\leftarrow "
             "\\\\ket{\\\\mathrm{this}} + \\\\ket{\\\\mathrm{state}}$.")
        .def("add_state_vector_with_coef",
             &StateVector::add_state_vector_with_coef,
             "add other state vector with multiplying the coef and make superposition. "
             "$\\\\ket{\\\\mathrm{this}}\\\\leftarrow\\\\ket{\\\\mathrm{this}}+\\\\mathrm{coef}"
             "\\\\ket{\\\\mathrm{"
             "state}}$.")
        .def("multiply_coef",
             &StateVector::multiply_coef,
             "Multiply coef. "
             "$\\\\ket{\\\\mathrm{this}}\\\\leftarrow\\\\mathrm{coef}\\\\ket{\\\\mathrm{this}}$.")
        .def(
            "sampling",
            [](const StateVector &state, UINT sampling_count, std::optional<UINT> seed) {
                return state.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            "Sampling specified times. Result is `list[int]` with the `sampling_count` length.")
        .def("to_string", &StateVector::to_string, "Information as `str`.")
        .def("load", &StateVector::load, "Load amplitudes of `list[int]` with `dim` length.")
        .def("__str__", &StateVector::to_string, "Information as `str`.")
        .def_ro_static("UNMEASURED",
                       &StateVector::UNMEASURED,
                       "Constant used for `StateVector::get_marginal_probability` to express the "
                       "the qubit is not measured.");

    nb::class_<StateVectorBatched>(
        m,
        "StateVectorBatched",
        "Batched vector representation of quantum state.\\n.. note:: Qubit index is start from 0. "
        "If the amplitudes of $\\\\ket{b_{n-1}\\\\dots b_0}$ is $b_i$, the state is $\\\\sum_i b_i "
        "2^i$.")
        .def(nb::init<UINT, UINT>(),
             "Construct batched state vector with specified batch size and qubits.")
        .def(nb::init<const StateVectorBatched &>(),
             "Constructing batched state vector by copying other batched state.")
        .def("n_qubits", &StateVectorBatched::n_qubits, "Get num of qubits.")
        .def("dim",
             &StateVectorBatched::dim,
             "Get dimension of the vector ($=2^\\\\mathrm{n\\\\_qubits}$).")
        .def("batch_size", &StateVectorBatched::batch_size, "Get batch size.")
        .def("set_state_vector",
             nb::overload_cast<const StateVector &>(&StateVectorBatched::set_state_vector),
             "Set the state vector for all batches.")
        .def("set_state_vector",
             nb::overload_cast<UINT, const StateVector &>(&StateVectorBatched::set_state_vector),
             "Set the state vector for a specific batch.")
        .def("get_state_vector",
             &StateVectorBatched::get_state_vector,
             "Get the state vector for a specific batch.")
        .def("set_zero_state",
             &StateVectorBatched::set_zero_state,
             "Initialize all batches with computational basis $\\\\ket{00\\\\dots0}$.")
        .def_static("Haar_random_states",
                    &StateVectorBatched::Haar_random_states,
                    "batch_size"_a,
                    "n_qubits"_a,
                    "seed"_a = std::random_device()(),
                    "Construct batched state vectors with Haar random states. If seed is not "
                    "specified, the value from random device is used.")
        .def("amplitudes",
             &StateVectorBatched::amplitudes,
             "Get all amplitudes with as `list[list[complex]]`.")
        .def("get_squared_norm",
             &StateVectorBatched::get_squared_norm,
             "Get squared norm of each state in the batch. $\\\\braket{\\\\psi|\\\\psi}$.")
        .def("normalize",
             &StateVectorBatched::normalize,
             "Normalize each state in the batch (let $\\\\braket{\\\\psi|\\\\psi} = 1$ by "
             "multiplying coef).")
        .def("get_zero_probability",
             &StateVectorBatched::get_zero_probability,
             "Get the probability to observe $\\\\ket{0}$ at specified index for each state in "
             "the batch.")
        .def("get_marginal_probability",
             &StateVectorBatched::get_marginal_probability,
             "Get the marginal probability to observe as specified for each state in the batch. "
             "Specify the result as n-length list. `0` and `1` represent the qubit is observed "
             "and get the value. `2` represents the qubit is not observed.")
        .def("get_entropy",
             &StateVectorBatched::get_entropy,
             "Get the entropy of each state in the batch.")
        .def("add_state_vector",
             &StateVectorBatched::add_state_vector,
             "Add other batched state vectors and make superposition. $\\\\ket{\\\\mathrm{this}} "
             "\\\\leftarrow \\\\ket{\\\\mathrm{this}} + \\\\ket{\\\\mathrm{states}}$.")
        .def("add_state_vector_with_coef",
             &StateVectorBatched::add_state_vector_with_coef,
             "Add other batched state vectors with multiplying the coef and make superposition. "
             "$\\\\ket{\\\\mathrm{this}}\\\\leftarrow\\\\ket{\\\\mathrm{this}}+\\\\mathrm{coef}"
             "\\\\ket{\\\\mathrm{states}}$.")
        .def("load",
             &StateVectorBatched::load,
             "Load batched amplitudes from `list[list[complex]]`.")
        .def("copy", &StateVectorBatched::copy, "Create a copy of the batched state vector.")
        .def("to_string", &StateVectorBatched::to_string, "Information as `str`.")
        .def("__str__", &StateVectorBatched::to_string, "Information as `str`.");

    nb::enum_<GateType>(m, "GateType", "Enum of Gate Type.")
        .value("I", GateType::I)
        .value("GlobalPhase", GateType::GlobalPhase)
        .value("X", GateType::X)
        .value("Y", GateType::Y)
        .value("Z", GateType::Z)
        .value("H", GateType::H)
        .value("S", GateType::S)
        .value("Sdag", GateType::Sdag)
        .value("T", GateType::T)
        .value("Tdag", GateType::Tdag)
        .value("SqrtX", GateType::SqrtX)
        .value("SqrtXdag", GateType::SqrtXdag)
        .value("SqrtY", GateType::SqrtY)
        .value("SqrtYdag", GateType::SqrtYdag)
        .value("P0", GateType::P0)
        .value("P1", GateType::P1)
        .value("RX", GateType::RX)
        .value("RY", GateType::RY)
        .value("RZ", GateType::RZ)
        .value("U1", GateType::U1)
        .value("U2", GateType::U2)
        .value("U3", GateType::U3)
        .value("OneQubitMatrix", GateType::OneQubitMatrix)
        .value("CX", GateType::CX)
        .value("CZ", GateType::CZ)
        .value("Swap", GateType::Swap)
        .value("TwoQubitMatrix", GateType::TwoQubitMatrix)
        .value("FusedSwap", GateType::FusedSwap)
        .value("Pauli", GateType::Pauli)
        .value("PauliRotation", GateType::PauliRotation);

#define DEF_GATE_BASE(GATE_TYPE, DESCRIPTION)                                            \
    nb::class_<GATE_TYPE>(m, #GATE_TYPE, DESCRIPTION)                                    \
        .def("gate_type", &GATE_TYPE::gate_type, "Get gate type as `GateType` enum.")    \
        .def(                                                                            \
            "get_target_qubit_list",                                                     \
            [](const GATE_TYPE &gate) { return gate->get_target_qubit_list(); },         \
            "Get target qubits as `list[int]`. **Control qubits is not included.**")     \
        .def(                                                                            \
            "get_control_qubit_list",                                                    \
            [](const GATE_TYPE &gate) { return gate->get_control_qubit_list(); },        \
            "Get control qubits as `list[int]`.")                                        \
        .def(                                                                            \
            "copy",                                                                      \
            [](const GATE_TYPE &gate) { return gate->copy(); },                          \
            "Copy gate as `Gate` type.")                                                 \
        .def(                                                                            \
            "get_inverse",                                                               \
            [](const GATE_TYPE &gate) { return gate->get_inverse(); },                   \
            "Generate inverse gate as `Gate` type. If not exists, return None.")         \
        .def(                                                                            \
            "update_quantum_state",                                                      \
            [](const GATE_TYPE &gate, StateVector &state_vector) {                       \
                gate->update_quantum_state(state_vector);                                \
            },                                                                           \
            "Apply gate to `state_vector`. `state_vector` in args is directly updated.") \
        .def(                                                                            \
            "get_matrix",                                                                \
            [](const GATE_TYPE &gate) { return gate->get_matrix(); },                    \
            "Get matrix representation of the gate. If cannot, None is returned.")

#define DEF_GATE(GATE_TYPE, DESCRIPTION)                                                       \
    DEF_GATE_BASE(                                                                             \
        GATE_TYPE,                                                                             \
        DESCRIPTION                                                                            \
        "\\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).") \
        .def(nb::init<Gate>())

    DEF_GATE_BASE(Gate,
                  "General class of QuantumGate.\\n.. note:: Downcast to requred to use "
                  "gate-specific functions.")
        .def(nb::init<IGate>(), "Upcast from `IGate`.")
        .def(nb::init<GlobalPhaseGate>(), "Upcast from `GlobalPhaseGate`.")
        .def(nb::init<XGate>(), "Upcast from `XGate`.")
        .def(nb::init<YGate>(), "Upcast from `YGate`.")
        .def(nb::init<ZGate>(), "Upcast from `ZGate`.")
        .def(nb::init<HGate>(), "Upcast from `HGate`.")
        .def(nb::init<SGate>(), "Upcast from `SGate`.")
        .def(nb::init<SdagGate>(), "Upcast from `SdagGate`.")
        .def(nb::init<TGate>(), "Upcast from `TGate`.")
        .def(nb::init<TdagGate>(), "Upcast from `TdagGate`.")
        .def(nb::init<SqrtXGate>(), "Upcast from `SqrtXGate`.")
        .def(nb::init<SqrtXdagGate>(), "Upcast from `SqrtXdagGate`.")
        .def(nb::init<SqrtYGate>(), "Upcast from `SqrtYGate`.")
        .def(nb::init<SqrtYdagGate>(), "Upcast from `SqrtYdagGate`.")
        .def(nb::init<P0Gate>(), "Upcast from `P0Gate`.")
        .def(nb::init<P1Gate>(), "Upcast from `P1Gate`.")
        .def(nb::init<RXGate>(), "Upcast from `RXGate`.")
        .def(nb::init<RYGate>(), "Upcast from `RYGate`.")
        .def(nb::init<RZGate>(), "Upcast from `RZGate`.")
        .def(nb::init<U1Gate>(), "Upcast from `U1Gate`.")
        .def(nb::init<U2Gate>(), "Upcast from `U2Gate`.")
        .def(nb::init<U3Gate>(), "Upcast from `U3Gate`.")
        .def(nb::init<OneQubitMatrixGate>(), "Upcast from `OneQubitMatrixGate`.")
        .def(nb::init<CXGate>(), "Upcast from `CXGate`.")
        .def(nb::init<CZGate>(), "Upcast from `CZGate`.")
        .def(nb::init<SwapGate>(), "Upcast from `SwapGate`.")
        .def(nb::init<TwoQubitMatrixGate>(), "Upcast from `TwoQubitMatrixGate`.")
        .def(nb::init<FusedSwapGate>(), "Upcast from `FusedSwapGate`.")
        .def(nb::init<PauliGate>(), "Upcast from `PauliGate`.")
        .def(nb::init<PauliRotationGate>(), "Upcast from `PauliRotationGate`.");

    DEF_GATE(IGate, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\\\mathrm{phase}}I$.")
        .def(
            "phase",
            [](const GlobalPhaseGate &gate) { return gate->phase(); },
            "Get `phase` property");

#define DEF_ONE_QUBIT_GATE(GATE_TYPE, DESCRIPTION)                             \
    DEF_GATE(GATE_TYPE, DESCRIPTION).def("target", [](const GATE_TYPE &gate) { \
        return gate->target();                                                 \
    })

    DEF_ONE_QUBIT_GATE(XGate, "Specific class of Pauli-X gate.");
    DEF_ONE_QUBIT_GATE(YGate, "Specific class of Pauli-Y gate.");
    DEF_ONE_QUBIT_GATE(ZGate, "Specific class of Pauli-Z gate.");
    DEF_ONE_QUBIT_GATE(HGate, "Specific class of Hadamard gate.");
    DEF_ONE_QUBIT_GATE(SGate,
                       "Specific class of S gate, represented as "
                       "$\\\\begin{bmatrix}\n1 & 0\\\\\\\\\n0 & i\n\\\\end{bmatrix}$.");
    DEF_ONE_QUBIT_GATE(SdagGate, "Specific class of inverse of S gate.");
    DEF_ONE_QUBIT_GATE(TGate,
                       "Specific class of T gate, represented as "
                       "$\\\\begin{bmatrix}\n1 & 0\\\\\\\\\n0 & e^{i\\\\pi/4}\n\\\\end{bmatrix}$.");
    DEF_ONE_QUBIT_GATE(TdagGate, "Specific class of inverse of T gate.");
    DEF_ONE_QUBIT_GATE(SqrtXGate,
                       "Specific class of sqrt(X) gate, represented as "
                       "$\\\\begin{bmatrix}\n1+i & 1-i\\\\\\\\\n1-i & 1+i\n\\\\end{bmatrix}$.");
    DEF_ONE_QUBIT_GATE(SqrtXdagGate, "Specific class of inverse of sqrt(X) gate.");
    DEF_ONE_QUBIT_GATE(SqrtYGate,
                       "Specific class of sqrt(Y) gate, represented as "
                       "$\\\\begin{bmatrix}\n1+i & -1-i \\\\\\\\\n 1+i & 1+i\n\\\\end{bmatrix}$.");
    DEF_ONE_QUBIT_GATE(SqrtYdagGate, "Specific class of inverse of sqrt(Y) gate.");
    DEF_ONE_QUBIT_GATE(
        P0Gate,
        "Specific class of projection gate to $\\\\ket{0}$.\\n.. note:: This gate is not unitary.");
    DEF_ONE_QUBIT_GATE(
        P1Gate,
        "Specific class of projection gate to $\\\\ket{1}$.\\n.. note:: This gate is not unitary.");

#define DEF_ONE_QUBIT_ROTATION_GATE(GATE_TYPE, DESCRIPTION) \
    DEF_ONE_QUBIT_GATE(GATE_TYPE, DESCRIPTION)              \
        .def(                                               \
            "angle", [](const GATE_TYPE &gate) { return gate->angle(); }, "Get `angle` property.")

    DEF_ONE_QUBIT_ROTATION_GATE(RXGate,
                                "Specific class of X rotation gate, represented as "
                                "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}X}$.");
    DEF_ONE_QUBIT_ROTATION_GATE(RYGate,
                                "Specific class of Y rotation gate, represented as "
                                "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}Y}$.");
    DEF_ONE_QUBIT_ROTATION_GATE(RZGate,
                                "Specific class of Z rotation gate, represented as "
                                "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}Z}$.");

    DEF_GATE(U1Gate,
             "Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, "
             "represented as "
             "$\\\\begin{bmatrix}\n1 & 0\\\\\\\\\n 0 & e^{i\\\\lambda}\n\\\\end{bmatrix}$.")
        .def(
            "lambda_", [](const U1Gate &gate) { return gate->lambda(); }, "Get `lambda` property.");
    DEF_GATE(U2Gate,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\\\frac{1}{\\\\sqrt{2}} \\\\begin{bmatrix}1 & -e^{-i\\\\lambda}\\\\\\\\\n "
             "e^{i\\\\phi} & e^{i(\\\\phi+\\\\lambda)}\n\\\\end{bmatrix}$.")
        .def(
            "phi", [](const U2Gate &gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_", [](const U2Gate &gate) { return gate->lambda(); }, "Get `lambda` property.");
    DEF_GATE(U3Gate,
             "Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, "
             "represented as "
             "$\\\\begin{bmatrix}\n\\\\cos \\\\frac{\\\\theta}{2} & "
             "-e^{i\\\\lambda}\\\\sin\\\\frac{\\\\theta}{2}\\\\\\\\\n "
             "e^{i\\\\phi}\\\\sin\\\\frac{\\\\theta}{2} & "
             "e^{i(\\\\phi+\\\\lambda)}\\\\cos\\\\frac{\\\\theta}{2}\n\\\\end{bmatrix}$.")
        .def(
            "theta", [](const U3Gate &gate) { return gate->theta(); }, "Get `theta` property.")
        .def(
            "phi", [](const U3Gate &gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_", [](const U3Gate &gate) { return gate->lambda(); }, "Get `lambda` property.");

    DEF_GATE(OneQubitMatrixGate, "Specific class of single-qubit dense matrix gate.")
        .def("matrix", [](const OneQubitMatrixGate &gate) { return gate->matrix(); });

#define DEF_ONE_CONTROL_ONE_TARGET_GATE(GATE_TYPE, DESCRIPTION)    \
    DEF_GATE(GATE_TYPE, DESCRIPTION)                               \
        .def(                                                      \
            "control",                                             \
            [](const GATE_TYPE &gate) { return gate->control(); }, \
            "Get property `control`.")                             \
        .def(                                                      \
            "target",                                              \
            [](const GATE_TYPE &gate) { return gate->target(); },  \
            "Get property `target`.")

    DEF_ONE_CONTROL_ONE_TARGET_GATE(CXGate,
                                    "Specific class of single-qubit-controlled Pauli-X gate.");
    DEF_ONE_CONTROL_ONE_TARGET_GATE(CZGate,
                                    "Specific class of single-qubit-controlled Pauli-Z gate.");

#define DEF_TWO_QUBIT_GATE(GATE_TYPE, DESCRIPTION)                 \
    DEF_GATE(GATE_TYPE, DESCRIPTION)                               \
        .def(                                                      \
            "target1",                                             \
            [](const GATE_TYPE &gate) { return gate->target1(); }, \
            "Get property `target1`.")                             \
        .def(                                                      \
            "target2",                                             \
            [](const GATE_TYPE &gate) { return gate->target2(); }, \
            "Get property `target2`.")

    DEF_TWO_QUBIT_GATE(SwapGate, "Specific class of two-qubit swap gate.");

    DEF_TWO_QUBIT_GATE(TwoQubitMatrixGate, "Specific class of double-qubit dense matrix gate.")
        .def(
            "matrix",
            [](const TwoQubitMatrixGate &gate) { gate->matrix(); },
            "Get property `matrix`.");

    DEF_GATE(
        FusedSwapGate,
        "Specific class of fused swap gate, which swap qubits in "
        "$[\\\\mathrm{qubit\\\\_index1},\\\\mathrm{qubit\\\\_index1}+\\\\mathrm{block\\\\_size})$ "
        "and qubits in "
        "$[\\\\mathrm{qubit\\\\_index2},\\\\mathrm{qubit\\\\_index2}+\\\\mathrm{block\\\\_size})$.")
        .def(
            "qubit_index1",
            [](const FusedSwapGate &gate) { return gate->qubit_index1(); },
            "Get property `qubit_index1`.")
        .def(
            "qubit_index2",
            [](const FusedSwapGate &gate) { return gate->qubit_index2(); },
            "Get property `qubit_index2`.")
        .def(
            "block_size",
            [](const FusedSwapGate &gate) { return gate->block_size(); },
            "Get property `block_size`.");

    DEF_GATE(PauliGate,
             "Specific class of multi-qubit pauli gate, which applies single-qubit Pauli "
             "gate to "
             "each of qubit.");
    DEF_GATE(PauliRotationGate,
             "Specific class of multi-qubit pauli-rotation gate, represented as "
             "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}P}$.");

#define DEF_GATE_FACTORY(GATE_NAME) \
    m.def(#GATE_NAME, &GATE_NAME, "Generate general Gate class instance of " #GATE_NAME ".")

    DEF_GATE_FACTORY(I);
    DEF_GATE_FACTORY(GlobalPhase);
    DEF_GATE_FACTORY(X);
    DEF_GATE_FACTORY(Y);
    DEF_GATE_FACTORY(Z);
    DEF_GATE_FACTORY(H);
    DEF_GATE_FACTORY(S);
    DEF_GATE_FACTORY(Sdag);
    DEF_GATE_FACTORY(T);
    DEF_GATE_FACTORY(Tdag);
    DEF_GATE_FACTORY(SqrtX);
    DEF_GATE_FACTORY(SqrtXdag);
    DEF_GATE_FACTORY(SqrtY);
    DEF_GATE_FACTORY(SqrtYdag);
    DEF_GATE_FACTORY(P0);
    DEF_GATE_FACTORY(P1);
    DEF_GATE_FACTORY(RX);
    DEF_GATE_FACTORY(RY);
    DEF_GATE_FACTORY(RZ);
    DEF_GATE_FACTORY(U1);
    DEF_GATE_FACTORY(U2);
    DEF_GATE_FACTORY(U3);
    DEF_GATE_FACTORY(CX);
    m.def(
        "CNot", &CX, "Generate general Gate class instance of CX.\n[note] CNot is an alias of CX.");
    DEF_GATE_FACTORY(CZ);
    DEF_GATE_FACTORY(Swap);
    DEF_GATE_FACTORY(FusedSwap);
    DEF_GATE_FACTORY(Pauli);
    DEF_GATE_FACTORY(PauliRotation);

    nb::enum_<ParamGateType>(m, "ParamGateType", "Enum of ParamGate Type.")
        .value("PRX", ParamGateType::PRX)
        .value("PRY", ParamGateType::PRY)
        .value("PRZ", ParamGateType::PRZ)
        .value("PPauliRotation", ParamGateType::PPauliRotation);

#define DEF_PGATE_BASE(PGATE_TYPE, DESCRIPTION)                                                   \
    nb::class_<PGATE_TYPE>(m, #PGATE_TYPE, DESCRIPTION)                                           \
        .def("param_gate_type",                                                                   \
             &PGATE_TYPE::param_gate_type,                                                        \
             "Get parametric gate type as `ParamGateType` enum.")                                 \
        .def(                                                                                     \
            "get_target_qubit_list",                                                              \
            [](const PGATE_TYPE &param_gate) { return param_gate->get_target_qubit_list(); },     \
            "Get target qubits as `list[int]`. **Control qubits is not included.**")              \
        .def(                                                                                     \
            "get_control_qubit_list",                                                             \
            [](const PGATE_TYPE &param_gate) { return param_gate->get_control_qubit_list(); },    \
            "Get control qubits as `list[int]`.")                                                 \
        .def(                                                                                     \
            "copy",                                                                               \
            [](const PGATE_TYPE &param_gate) { return param_gate->copy(); },                      \
            "Copy gate as `ParamGate` type.")                                                     \
        .def(                                                                                     \
            "get_inverse",                                                                        \
            [](const PGATE_TYPE &param_gate) { return param_gate->get_inverse(); },               \
            "Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.")  \
        .def(                                                                                     \
            "update_quantum_state",                                                               \
            [](const PGATE_TYPE &param_gate, StateVector &state_vector, double param) {           \
                param_gate->update_quantum_state(state_vector, param);                            \
            },                                                                                    \
            "Apply gate to `state_vector` with holding the parameter. `state_vector` in args is " \
            "directly updated.")                                                                  \
        .def(                                                                                     \
            "get_matrix",                                                                         \
            [](const PGATE_TYPE &gate, double param) { return gate->get_matrix(param); },         \
            "Get matrix representation of the gate with holding the parameter. If cannot, None "  \
            "is returned.")

#define DEF_PGATE(PGATE_TYPE, DESCRIPTION)                                                     \
    DEF_PGATE_BASE(                                                                            \
        PGATE_TYPE,                                                                            \
        DESCRIPTION                                                                            \
        "\\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).") \
        .def(nb::init<ParamGate>())

    DEF_PGATE_BASE(
        ParamGate,
        "General class of parametric quantum gate.\\n.. note:: Downcast to requred to use "
        "gate-specific functions.")
        .def(nb::init<PRXGate>(), "Upcast from `PRXGate`.")
        .def(nb::init<PRYGate>(), "Upcast from `PRYGate`.")
        .def(nb::init<PRZGate>(), "Upcast from `PRZGate`.")
        .def(nb::init<PPauliRotationGate>(), "Upcast from `PPauliRotationGate`.");

#define DEF_ONE_QUBIT_PGATE(PGATE_TYPE, DESCRIPTION)                                    \
    DEF_PGATE(PGATE_TYPE, DESCRIPTION).def("target", [](const PGATE_TYPE &param_gate) { \
        return param_gate->target();                                                    \
    })

    DEF_ONE_QUBIT_PGATE(
        PRXGate,
        "Specific class of parametric X rotation gate, represented as "
        "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}X}$. `angle` is given as `param * pcoef`.");
    DEF_ONE_QUBIT_PGATE(
        PRYGate,
        "Specific class of parametric Y rotation gate, represented as "
        "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}Y}$. `angle` is given as `param * pcoef`.");
    DEF_ONE_QUBIT_PGATE(
        PRZGate,
        "Specific class of parametric Z rotation gate, represented as "
        "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}Z}$. `angle` is given as `param * pcoef`.");

    DEF_PGATE(PPauliRotationGate,
              "Specific class of parametric multi-qubit pauli-rotation gate, represented as "
              "$e^{-i\\\\frac{\\\\mathrm{angle}}{2}P}$. `angle` is given as `param * pcoef`.");

#define DEF_PGATE_FACTORY(PGATE_NAME) \
    m.def(#PGATE_NAME, &PGATE_NAME, "Generate general ParamGate class instance of " #PGATE_NAME ".")

    m.def("PRX",
          &PRX,
          "Generate general ParamGate class instance of PRX.",
          "target"_a,
          "coef"_a = 1.);
    m.def("PRY",
          &PRY,
          "Generate general ParamGate class instance of PRY.",
          "target"_a,
          "coef"_a = 1.);
    m.def("PRZ",
          &PRZ,
          "Generate general ParamGate class instance of PRZ.",
          "target"_a,
          "coef"_a = 1.);
    m.def("PPauliRotation",
          &PPauliRotation,
          "Generate general ParamGate class instance of PPauliRotation.",
          "pauli"_a,
          "coef"_a = 1.);

    nb::class_<Circuit>(m, "Circuit", "Quantum circuit represented as gate array")
        .def(nb::init<UINT>(), "Initialize empty circuit of specified qubits.")
        .def("n_qubits", &Circuit::n_qubits, "Get property of `n_qubits`.")
        .def("gate_list", &Circuit::gate_list, "Get property of `gate_list`.")
        .def("gate_count", &Circuit::gate_count, "Get property of `gate_count`.")
        .def("key_set", &Circuit::key_set, "Get set of keys of parameters.")
        .def("get", &Circuit::get, "Get reference of i-th gate.")
        .def("get_key",
             &Circuit::get_key,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate &>(&Circuit::add_gate),
             "Add gate. Given gate is copied.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate &, std::string_view>(&Circuit::add_param_gate),
             "Add parametric gate with specifing key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit &>(&Circuit::add_circuit),
             "Add all gates in specified circuit. Given gates are copied.")
        .def("update_quantum_state",
             &Circuit::update_quantum_state,
             "Apply gate to the StateVector. StateVector in args is directly updated. If the "
             "circuit contains parametric gate, you have to give real value of parameter as "
             "dict[str, float] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit &circuit, StateVector &state, nb::kwargs kwargs) {
                std::map<std::string, double> parameters;
                for (auto &&[key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
                }
                circuit.update_quantum_state(state, parameters);
            },
            "Apply gate to the StateVector. StateVector in args is directly updated. If the "
            "circuit contains parametric gate, you have to give real value of parameter as "
            "\"name=value\" format in kwargs.")
        .def(
            "update_quantum_state",
            [](const Circuit &circuit, StateVector &state) { circuit.update_quantum_state(state); })
        .def("copy", &Circuit::copy, "Copy circuit. All the gates inside is copied.")
        .def("get_inverse",
             &Circuit::get_inverse,
             "Get inverse of circuit. ALl the gates are newly created.");

    nb::class_<PauliOperator>(
        m,
        "PauliOperator",
        "Pauli operator as coef and tensor product of single pauli for each qubit.")
        .def(nb::init<Complex>(), "coef"_a = 1., "Initialize operator which just multiplying coef.")
        .def(nb::init<const std::vector<UINT> &, const std::vector<UINT> &, Complex>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`pauli_id_list[i]` is "
             "applied to `target_qubit_list`-th qubit.")
        .def(nb::init<std::string_view, Complex>(),
             "pauli_string"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. If `pauli_string` is `\"X0Y2\"`, Pauli-X is applied to "
             "0-th "
             "qubit and Pauli-Y is applied to 2-th qubit. In `pauli_string`, spaces are ignored.")
        .def(nb::init<const std::vector<UINT> &, Complex>(),
             "pauli_id_par_qubit"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`paul_id_per_qubit` is "
             "applied to `i`-th qubit.")
        .def(
            "__init__",
            [](PauliOperator *t,
               nb::int_ bit_flip_mask_py,
               nb::int_ phase_flip_mask_py,
               Complex coef) {
                internal::BitVector bit_flip_mask(0), phase_flip_mask(0);
                const nb::int_ mask(~0ULL);
                auto &bit_flip_raw = bit_flip_mask.data_raw();
                assert(bit_flip_raw.empty());
                while (bit_flip_mask_py > nb::int_(0)) {
                    bit_flip_raw.push_back((UINT)nb::int_(bit_flip_mask_py & mask));
                    bit_flip_mask_py >>= nb::int_(64);
                }
                auto &phase_flip_raw = phase_flip_mask.data_raw();
                assert(phase_flip_raw.empty());
                while (phase_flip_mask_py > nb::int_(0)) {
                    phase_flip_raw.push_back((UINT)nb::int_(phase_flip_mask_py & mask));
                    phase_flip_mask_py >>= nb::int_(64);
                }
                new (t) PauliOperator(bit_flip_mask, phase_flip_mask, coef);
            },
            "bit_flip_mask"_a,
            "phase_flip_mask"_a,
            "coef"_a = 1.,
            "Initialize pauli operator. For each `i`, single pauli applied to `i`-th qubit is got "
            "from `i-th` bit of `bit_flip_mask` and `phase_flip_mask` as "
            "follows.\\n|bit_flip|phase_flip|pauli|\\n|--|--|--|\\n|0|0|I|\\n|0|1|Z|\\n|1|0|X|\\n|"
            "1|1|Y|")
        .def("get_coef", &PauliOperator::get_coef, "Get property `coef`.")
        .def("get_target_qubit_list",
             &PauliOperator::get_target_qubit_list,
             "Get qubits to be applied pauli.")
        .def("get_pauli_id_list",
             &PauliOperator::get_pauli_id_list,
             "Get pauli id to be applied. The order is correspond to the result of "
             "`get_target_qubit_list`")
        .def(
            "get_XZ_mask_representation",
            [](const PauliOperator &pauli) {
                const auto [x_mask, z_mask] = pauli.get_XZ_mask_representation();
                nb::int_ x_mask_py(0);
                for (UINT i = 0; i < x_mask.size(); ++i) {
                    x_mask_py |= nb::int_(x_mask[i]) << nb::int_(i);
                }
                nb::int_ z_mask_py(0);
                for (UINT i = 0; i < z_mask.size(); ++i) {
                    z_mask_py |= nb::int_(z_mask[i]) << nb::int_(i);
                }
                return std::make_tuple(x_mask_py, z_mask_py);
            },
            "Get single-pauli property as binary integer representation. See description of "
            "`__init__(bit_flip_mask_py: int, phase_flip_mask_py: int, coef: float=1.)` for "
            "details.")
        .def("get_pauli_string",
             &PauliOperator::get_pauli_string,
             "Get single-pauli property as string representation. See description of "
             "`__init__(pauli_string: str, coef: float=1.)` for details.")
        .def("get_dagger", &PauliOperator::get_dagger, "Get adjoint operator.")
        .def("get_qubit_count",
             &PauliOperator::get_qubit_count,
             "Get num of qubits to applied with, when count from 0-th qubit. Subset of $[0, "
             "\\\\mathrm{qubit_count})$ is the "
             "target.")
        .def("change_coef", &PauliOperator::change_coef, "Set property `coef`.")
        .def("add_single_pauli",
             &PauliOperator::add_single_pauli,
             "Add (apply tensor product) another single pauli. You cannot specify qubit index that "
             "has "
             "always a single "
             "pauli.")
        .def("apply_to_state", &PauliOperator::apply_to_state, "Apply pauli to state vector.")
        .def(
            "get_expectation_value",
            &PauliOperator::get_expectation_value,
            "Get expectation value of measuring state vector. $\\\\bra{\\\\psi}P\\\\ket{\\\\psi}$.")
        .def("get_transition_amplitude",
             &PauliOperator::get_transition_amplitude,
             "Get transition amplitude of measuring state vector. "
             "$\\\\bra{\\\\chi}P\\\\ket{\\\\psi}$.")
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self *= Complex())
        .def(nb::self * Complex())
        .def_ro_static("I", &PauliOperator::I)
        .def_ro_static("X", &PauliOperator::X)
        .def_ro_static("Y", &PauliOperator::Y)
        .def_ro_static("Z", &PauliOperator::Z);

    nb::class_<Operator>(m, "Operator")
        .def(nb::init<UINT>())
        .def("is_hermitian", &Operator::is_hermitian)
        .def("n_qubits", &Operator::n_qubits)
        .def("terms", &Operator::terms)
        .def("to_string", &Operator::to_string)
        .def("add_operator", nb::overload_cast<const PauliOperator &>(&Operator::add_operator))
        .def(
            "add_random_operator",
            [](Operator &op, UINT operator_count, std::optional<UINT> seed) {
                return op.add_random_operator(operator_count,
                                              seed.value_or(std::random_device{}()));
            },
            "operator_count"_a,
            "seed"_a = std::nullopt)
        .def("optimize", &Operator::optimize)
        .def("get_dagger", &Operator::get_dagger)
        .def("apply_to_state", &Operator::apply_to_state)
        .def("get_expectation_value", &Operator::get_expectation_value)
        .def("get_transition_amplitude", &Operator::get_transition_amplitude)
        .def(nb::self *= Complex())
        .def(nb::self * Complex())
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self += PauliOperator())
        .def(nb::self + PauliOperator())
        .def(nb::self -= PauliOperator())
        .def(nb::self - PauliOperator())
        .def(nb::self *= PauliOperator())
        .def(nb::self * PauliOperator());
}
