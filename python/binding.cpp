#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
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

NB_MODULE(scaluq_core, m) {
    nb::class_<InitializationSettings>(
        m,
        "InitializationSettings",
        "Wrapper class of Kokkos's InitializationSettings.\nSee details: "
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
    m.def(
        "finalize", &finalize, "Terminate the Kokkos execution environment. Release the resources");
    m.def("is_initialized", &is_initialized, "Return true if initialize() is already called".);
    m.def("is_finalized", &is_initialized, "Return true if finalize() is already called.");

    nb::class_<StateVector>(
        m,
        "StateVector",
        "Vector representation of quantum state.\n[note] Qubit index is "
        "start from 0. The amplitudes that ith qubit is b_i ∈ {0, 1} has an index of Σ(b_i 2^i).")
        .def(nb::init<UINT>(),
             "Construct state vector with specified qubits, initialized with computational "
             "basis |0...0>.")
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
        .def("get_amplitude_at_index",
             &StateVector::get_amplitude_at_index,
             "Get amplitude at one index.\n[note] If you want to get all amplitudes, you should "
             "use StateVector::amplitudes().")
        .def("set_zero_state",
             &StateVector::set_zero_state,
             "Initialize with computational basis |00...0>.")
        .def("set_zero_norm_state",
             &StateVector::set_zero_norm_state,
             "Initialize with 0 (null vector).")
        .def("set_computational_basis",
             &StateVector::set_computational_basis,
             "Initialize with computational basis |basis>.")
        .def("amplitudes", &StateVector::amplitudes, "Get all amplitudes with as List[complex].")
        .def("n_qubits", &StateVector::n_qubits, "Get num of qubits.")
        .def("dim", &StateVector::dim, "Get dimension of the vector (=2^`n_qubits`).")
        .def("get_squared_norm",
             &StateVector::get_squared_norm,
             "Get squared norm of the state. <ψ|ψ>.")
        .def("normalize",
             &StateVector::normalize,
             "Normalize state (let <ψ|ψ> = 1 by multiplying coef).")
        .def("get_zero_probability",
             &StateVector::get_zero_probability,
             "Get the probability to observe |0> at specified index.")
        .def("get_marginal_probability",
             &StateVector::get_marginal_probability,
             "Get the marginal probability to observe as specified. Specify the result as n-length "
             "list. `0` and `1` represent the qubit is observed and get the value. `2` represents "
             "the qubit is not observed.")
        .def("get_entropy", &StateVector::get_entropy, "Get the entropy of the vector.")
        .def("add_state_vector",
             &StateVector::add_state_vector,
             "Add other state vector and make superposition. += |`state`>.")
        .def("add_state_vector_with_coef",
             &StateVector::add_state_vector_with_coef,
             "add other state vector with multiplying the coef and make superposition. += "
             "`coef`|`state`>".)
        .def("multiply_coef", &StateVector::multiply_coef, "Multiply coef.")
        .def(
            "sampling",
            [](const StateVector &state, UINT sampling_count, std::optional<UINT> seed) {
                return state.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            "Sampling specified times. Result is list[int] with the `sampling_count` length.")
        .def("to_string", &StateVector::to_string, "Information as str.")
        .def("load", &StateVector::load, "Load amplitudes of List[int] with `dim` length.")
        .def("__str__", &StateVector::to_string, "Information as str.");

    nb::enum_<GateType>(m, "GateType", "enum of Gate Type")
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

#define DEF_GATE_BASE(GATE_TYPE, DESCRIPTION)                                                      \
    nb::class_<GATE_TYPE>(m, #GATE_TYPE, DESCRIPTION)                                              \
        .def("gate_type", &GATE_TYPE::gate_type, "Get gate_type as GateType enum.")                \
        .def(                                                                                      \
            "get_target_qubit_list",                                                               \
            [](const GATE_TYPE &gate) { return gate->get_target_qubit_list(); },                   \
            "Get target qubits as List[int] (control qubits is not included).")                    \
        .def(                                                                                      \
            "get_control_qubit_list",                                                              \
            [](const GATE_TYPE &gate) { return gate->get_control_qubit_list(); },                  \
            "Get control qubits as List[int].")                                                    \
        .def(                                                                                      \
            "copy", [](const GATE_TYPE &gate) { return gate->copy(); }, "Copy gate as Gate type.") \
        .def(                                                                                      \
            "get_inverse",                                                                         \
            [](const GATE_TYPE &gate) { return gate->get_inverse(); },                             \
            "Generate inverse gate as Gate type. If not exists, return None.")                     \
        .def(                                                                                      \
            "update_quantum_state",                                                                \
            [](const GATE_TYPE &gate, StateVector &state_vector) {                                 \
                gate->update_quantum_state(state_vector);                                          \
            },                                                                                     \
            "Apply gate to the StateVector. StateVector in args is directly updated.")

#define DEF_GATE(GATE_TYPE, DESCRIPTION)                                                       \
    DEF_GATE_BASE(                                                                             \
        GATE_TYPE,                                                                             \
        (std::string)DESCRIPTION +                                                             \
            "\n[note] Upcast is required to use gate-general functions (ex: add to Circuit).") \
        .def(nb::init<Gate>())

    DEF_GATE_BASE(
        Gate,
        "General class of QuantumGate.\n[note] Downcast to requred to use gate-specific functions.")
        .def(nb::init<IGate>(), "Upcast from IGate.")
        .def(nb::init<GlobalPhaseGate>(), "Upcast from GlobalPhaseGate.")
        .def(nb::init<XGate>(), "Upcast from XGate.")
        .def(nb::init<YGate>(), "Upcast from YGate.")
        .def(nb::init<ZGate>(), "Upcast from ZGate.")
        .def(nb::init<HGate>(), "Upcast from HGate.")
        .def(nb::init<SGate>(), "Upcast from SGate.")
        .def(nb::init<SdagGate>(), "Upcast from SdagGate.")
        .def(nb::init<TGate>(), "Upcast from TGate.")
        .def(nb::init<TdagGate>(), "Upcast from TdagGate.")
        .def(nb::init<SqrtXGate>(), "Upcast from SqrtXGate.")
        .def(nb::init<SqrtXdagGate>(), "Upcast from SqrtXdagGate.")
        .def(nb::init<SqrtYGate>(), "Upcast from SqrtYGate.")
        .def(nb::init<SqrtYdagGate>(), "Upcast from SqrtYdagGate.")
        .def(nb::init<P0Gate>(), "Upcast from P0Gate.")
        .def(nb::init<P1Gate>(), "Upcast from P1Gate.")
        .def(nb::init<RXGate>(), "Upcast from RXGate.")
        .def(nb::init<RYGate>(), "Upcast from RYGate.")
        .def(nb::init<RZGate>(), "Upcast from RZGate.")
        .def(nb::init<U1Gate>(), "Upcast from U1Gate.")
        .def(nb::init<U2Gate>(), "Upcast from U2Gate.")
        .def(nb::init<U3Gate>(), "Upcast from U3Gate.")
        .def(nb::init<OneQubitMatrixGate>(), "Upcast from OneQubitMatrixGate.")
        .def(nb::init<CXGate>(), "Upcast from CXGate.")
        .def(nb::init<CZGate>(), "Upcast from CZGate.")
        .def(nb::init<SwapGate>(), "Upcast from SwapGate.")
        .def(nb::init<TwoQubitMatrixGate>(), "Upcast from TwoQubitMatrixGate.")
        .def(nb::init<FusedSwapGate>(), "Upcast from FusedSwapGate.")
        .def(nb::init<PauliGate>(), "Upcast from PauliGate.")
        .def(nb::init<PauliRotationGate>()),
        "Upcast from PauliRotationGate.";

    DEF_GATE(IGate, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             "Specific class of gate, which rotate global phase, represented as e^(i `phase`) I.")
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
    DEF_ONE_QUBIT_GATE(SGate, "Specific class of S gate, represented as [[1, 0], [0, i]].");
    DEF_ONE_QUBIT_GATE(SdagGate, "Specific class of inverse of S gate.");
    DEF_ONE_QUBIT_GATE(TGate, "Specific class of T gate, represented as [[1, 0], [0, e^(iπ/4)]].");
    DEF_ONE_QUBIT_GATE(TdagGate, "Specific class of inverse of T gate.");
    DEF_ONE_QUBIT_GATE(SqrtXGate,
                       "Specific class of sqrt(X) gate, represented as [[1+i, 1-i], [1-i, 1+i]].");
    DEF_ONE_QUBIT_GATE(SqrtXdagGate, "Specific class of inverse of sqrt(X) gate.");
    DEF_ONE_QUBIT_GATE(SqrtYGate,
                       "Specific class of sqrt(Y) gate, represented as [[1+i, -1-i], [1+i, 1+i]].");
    DEF_ONE_QUBIT_GATE(SqrtYdagGate, "Specific class of inverse of sqrt(Y) gate.");
    DEF_ONE_QUBIT_GATE(
        P0Gate, "Specific class of projection gate to |0>.\n[note] This gate is not unitary.");
    DEF_ONE_QUBIT_GATE(
        P1Gate, "Specific class of projection gate to |1>.\n[note] This gate is not unitary.");

#define DEF_ONE_QUBIT_ROTATION_GATE(GATE_TYPE, DESCRIPTION) \
    DEF_ONE_QUBIT_GATE(GATE_TYPE, DESCRIPTION)              \
        .def(                                               \
            "angle", [](const GATE_TYPE &gate) { return gate->angle(); }, "Get `angle` property.")

    DEF_ONE_QUBIT_ROTATION_GATE(
        RXGate, "Specific class of X rotation gate, represented as e^(-i(`angle`/2)X).");
    DEF_ONE_QUBIT_ROTATION_GATE(
        RYGate, "Specific class of Y rotation gate, represented as e^(-i(`angle`/2)Y).");
    DEF_ONE_QUBIT_ROTATION_GATE(
        RZGate, "Specific class of Z rotation gate, represented as e^(-i(`angle`/2)Z).");

    DEF_GATE(U1Gate,
             "Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, represented as "
             "[[1, 0], [0, e^(i`lambda`)]].")
        .def(
            "lambda_", [](const U1Gate &gate) { return gate->lambda(); }, "Get `lambda` property.");
    DEF_GATE(U2Gate,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, represented as "
             "(1/sqrt(2)) [[1, -e^(-i`lambda`)], [e^(i`phi`), e^(i(`phi`+`lambda`))]].")
        .def(
            "phi", [](const U2Gate &gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_", [](const U2Gate &gate) { return gate->lambda(); }, "Get `lambda` property.");
    DEF_GATE(U3Gate,
             "Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, represented as "
             "[[cos(`theta`/2), -e^(i`lambda`)sin(`theta`/2)], [e^(i`phi`)sin(`theta`/2)], "
             "e^(i(`phi`+`lambda`))cos(`theta`/2)].")
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

    DEF_GATE(FusedSwapGate,
             "Specific class of fused swap gate, which swap qubits in "
             "[`qubit_index1`..`qubit_index1+block_size`) and qubits in "
             "[`qubit_index2`..`qubit_index2`+block_size`).")
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
             "Specific class of multi-qubit pauli gate, which applies single-qubit Pauli gate to "
             "each of qubit");
    DEF_GATE(
        PauliRotationGate,
        "Specific class of multi-qubit pauli-rotation gate, represented as e^(-i(`angle`/2)P).");

#define DEF_GATE_FACTORY(GATE_NAME) \
    m.def(#GATE_NAME, &GATE_NAME, "Generate general Gate class instance of "s + #GATE_NAME)

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

    nb::class_<Circuit>(m, "Circuit")
        .def(nb::init<UINT>())
        .def("n_qubits", &Circuit::n_qubits)
        .def("gate_list", &Circuit::gate_list)
        .def("gate_count", &Circuit::gate_count)
        .def("get", nb::overload_cast<UINT>(&Circuit::get))
        .def("calculate_depth", &Circuit::calculate_depth)
        .def("add_gate", nb::overload_cast<const Gate &>(&Circuit::add_gate))
        .def("add_circuit", nb::overload_cast<const Circuit &>(&Circuit::add_circuit))
        .def("update_quantum_state", &Circuit::update_quantum_state)
        .def("copy", &Circuit::copy)
        .def("get_inverse", &Circuit::get_inverse);

    nb::class_<PauliOperator>(m, "PauliOperator")
        .def(nb::init<Complex>(), "coef"_a = 1.)
        .def(nb::init<const std::vector<UINT> &, const std::vector<UINT> &, Complex>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.)
        .def(nb::init<std::string_view, Complex>(), "pauli_string"_a, "coef"_a = 1.)
        .def(nb::init<const std::vector<UINT> &, Complex>(), "pauli_id_par_qubit"_a, "coef"_a = 1.)
        .def(
            "__init__",
            [](PauliOperator *t,
               nb::int_ bit_flip_mask_py,
               nb::int_ phase_flip_mask_py,
               Complex coef) {
                BitVector bit_flip_mask(0), phase_flip_mask(0);
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
            "coef"_a = 1.)
        .def("get_coef", &PauliOperator::get_coef)
        .def("get_target_qubit_list", &PauliOperator::get_target_qubit_list)
        .def("get_pauli_id_list", &PauliOperator::get_pauli_id_list)
        .def("get_XZ_mask_representation",
             [](const PauliOperator &pauli) {
                 const auto &[x_mask, z_mask] = pauli.get_XZ_mask_representation();
                 const auto &x_raw = x_mask.data_raw();
                 nb::int_ x_mask_py(0);
                 for (UINT i = 0; i < x_raw.size(); ++i) {
                     x_mask_py |= nb::int_(x_raw[i]) << nb::int_(64 * i);
                 }
                 const auto &z_raw = z_mask.data_raw();
                 nb::int_ z_mask_py(0);
                 for (UINT i = 0; i < z_raw.size(); ++i) {
                     z_mask_py |= nb::int_(z_raw[i]) << nb::int_(64 * i);
                 }
                 return std::make_tuple(x_mask_py, z_mask_py);
             })
        .def("get_pauli_string", &PauliOperator::get_pauli_string)
        .def("get_dagger", &PauliOperator::get_dagger)
        .def("get_qubit_count", &PauliOperator::get_qubit_count)
        .def("change_coef", &PauliOperator::change_coef)
        .def("add_single_pauli", &PauliOperator::add_single_pauli)
        .def("apply_to_state", &PauliOperator::apply_to_state)
        .def("get_expectation_value", &PauliOperator::get_expectation_value)
        .def("get_transition_amplitude", &PauliOperator::get_transition_amplitude)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self *= Complex())
        .def(nb::self * Complex());

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
