#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <all.hpp>

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

    m.def("initialize", &initialize, "settings"_a = InitializationSettings());
    m.def("finalize", &finalize);

    nb::class_<StateVector>(m, "StateVector")
        .def(nb::init<>())
        .def(nb::init<UINT>())
        .def(nb::init<const StateVector &>())
        .def_static(
            "Haar_random_state",
            [](UINT n_qubits, std::optional<UINT> seed) {
                return StateVector::Haar_random_state(n_qubits,
                                                      seed.value_or(std::random_device{}()));
            },
            "n_qubits"_a,
            "seed"_a = std::nullopt)
        .def("set_amplitude_at_index", &StateVector::set_amplitude_at_index)
        .def("get_amplitude_at_index", &StateVector::get_amplitude_at_index)
        .def("set_zero_state", &StateVector::set_zero_state)
        .def("set_zero_norm_state", &StateVector::set_zero_norm_state)
        .def("set_computational_basis", &StateVector::set_computational_basis)
        .def("amplitudes", &StateVector::amplitudes)
        .def("n_qubits", &StateVector::n_qubits)
        .def("dim", &StateVector::dim)
        .def("get_squared_norm", &StateVector::get_squared_norm)
        .def("normalize", &StateVector::normalize)
        .def("get_zero_probability", &StateVector::get_zero_probability)
        .def("get_marginal_probability", &StateVector::get_marginal_probability)
        .def("get_entropy", &StateVector::get_entropy)
        .def("add_state_vector", &StateVector::add_state_vector)
        .def("add_state_vector_with_coef", &StateVector::add_state_vector_with_coef)
        .def("multiply_coef", &StateVector::multiply_coef)
        .def(
            "sampling",
            [](const StateVector &state, UINT sampling_count, std::optional<UINT> seed) {
                return state.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt)
        .def("to_string", &StateVector::to_string)
        .def("load", &StateVector::load)
        .def("__str__", &StateVector::to_string);

#define DEF_GATE(GATE_TYPE)                                                                 \
    nb::class_<GATE_TYPE>(m, #GATE_TYPE)                                                    \
        .def(nb::init<Gate>())                                                              \
        .def("get_target_qubit_list",                                                       \
             [](const GATE_TYPE &gate) { return gate->get_target_qubit_list(); })           \
        .def("get_control_qubit_list",                                                      \
             [](const GATE_TYPE &gate) { return gate->get_control_qubit_list(); })          \
        .def("copy", [](const GATE_TYPE &gate) { return gate->copy(); })                    \
        .def("get_inverse", [](const GATE_TYPE &gate) { return gate->get_inverse(); })      \
        .def("update_quantum_state", [](const GATE_TYPE &gate, StateVector &state_vector) { \
            gate->update_quantum_state(state_vector);                                       \
        })

    DEF_GATE(Gate);

#define DEF_ONE_QUBIT_GATE(GATE_TYPE) \
    DEF_GATE(GATE_TYPE).def("target", [](const GATE_TYPE &gate) { return gate->target(); })

    DEF_ONE_QUBIT_GATE(IGate);
    DEF_ONE_QUBIT_GATE(XGate);
    DEF_ONE_QUBIT_GATE(YGate);
    DEF_ONE_QUBIT_GATE(ZGate);
    DEF_ONE_QUBIT_GATE(HGate);
    DEF_ONE_QUBIT_GATE(SGate);
    DEF_ONE_QUBIT_GATE(SdagGate);
    DEF_ONE_QUBIT_GATE(TGate);
    DEF_ONE_QUBIT_GATE(TdagGate);
    DEF_ONE_QUBIT_GATE(sqrtXGate);
    DEF_ONE_QUBIT_GATE(sqrtXdagGate);
    DEF_ONE_QUBIT_GATE(sqrtYGate);
    DEF_ONE_QUBIT_GATE(sqrtYdagGate);
    DEF_ONE_QUBIT_GATE(P0Gate);
    DEF_ONE_QUBIT_GATE(P1Gate);

#define DEF_ONE_QUBIT_ROTATION_GATE(GATE_TYPE) \
    DEF_ONE_QUBIT_GATE(GATE_TYPE).def("angle", [](const GATE_TYPE &gate) { return gate->angle(); })

    DEF_ONE_QUBIT_ROTATION_GATE(RXGate);
    DEF_ONE_QUBIT_ROTATION_GATE(RYGate);
    DEF_ONE_QUBIT_ROTATION_GATE(RZGate);

    DEF_GATE(U1Gate).def("lambda_", [](const U1Gate &gate) { return gate->lambda(); });
    DEF_GATE(U2Gate)
        .def("phi", [](const U2Gate &gate) { return gate->phi(); })
        .def("lambda_", [](const U2Gate &gate) { return gate->lambda(); });
    DEF_GATE(U3Gate)
        .def("theta", [](const U3Gate &gate) { return gate->theta(); })
        .def("phi", [](const U3Gate &gate) { return gate->phi(); })
        .def("lambda_", [](const U3Gate &gate) { return gate->lambda(); });

#define DEF_ONE_CONTROL_ONE_TARGET_GATE(GATE_TYPE)                             \
    DEF_GATE(GATE_TYPE)                                                        \
        .def("control", [](const GATE_TYPE &gate) { return gate->control(); }) \
        .def("target", [](const GATE_TYPE &gate) { return gate->target(); })

    DEF_ONE_CONTROL_ONE_TARGET_GATE(CNOTGate);
    DEF_ONE_CONTROL_ONE_TARGET_GATE(CZGate);

    DEF_GATE(SWAPGate)
        .def("target1", [](const SWAPGate &gate) { return gate->target1(); })
        .def("target2", [](const SWAPGate &gate) { return gate->target2(); });

    DEF_GATE(FusedSWAPGate)
        .def("qubit_index1", [](const FusedSWAPGate &gate) { return gate->qubit_index1(); })
        .def("qubit_index2", [](const FusedSWAPGate &gate) { return gate->qubit_index2(); })
        .def("block_size", [](const FusedSWAPGate &gate) { return gate->block_size(); });

#define DEF_GATE_FACTORY(GATE_NAME) m.def(#GATE_NAME, &GATE_NAME)

    DEF_GATE_FACTORY(I);
    DEF_GATE_FACTORY(X);
    DEF_GATE_FACTORY(Y);
    DEF_GATE_FACTORY(Z);
    DEF_GATE_FACTORY(H);
    DEF_GATE_FACTORY(S);
    DEF_GATE_FACTORY(Sdag);
    DEF_GATE_FACTORY(T);
    DEF_GATE_FACTORY(Tdag);
    DEF_GATE_FACTORY(sqrtX);
    DEF_GATE_FACTORY(sqrtXdag);
    DEF_GATE_FACTORY(sqrtY);
    DEF_GATE_FACTORY(sqrtYdag);
    DEF_GATE_FACTORY(P0);
    DEF_GATE_FACTORY(P1);
    DEF_GATE_FACTORY(RX);
    DEF_GATE_FACTORY(RY);
    DEF_GATE_FACTORY(RZ);
    DEF_GATE_FACTORY(U1);
    DEF_GATE_FACTORY(U2);
    DEF_GATE_FACTORY(U3);
    DEF_GATE_FACTORY(CNOT);
    DEF_GATE_FACTORY(CZ);
    DEF_GATE_FACTORY(SWAP);
    DEF_GATE_FACTORY(FusedSWAP);

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
