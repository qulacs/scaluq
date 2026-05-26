#pragma once

#include <bit>
#include <map>
#include <random>
#include <set>
#include <string_view>
#include <variant>

#include "../classical_register/classical_register.hpp"
#include "../classical_register/classical_register_batched.hpp"
#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../operator/operator.hpp"
#include "../types.hpp"

namespace scaluq {

template <Precision Prec>
class Circuit {
public:
    using GateWithKey = std::variant<Gate<Prec>, std::pair<ParamGate<Prec>, std::string>>;
    Circuit() = default;

    [[nodiscard]] inline const std::vector<GateWithKey>& gate_list() const { return _gate_list; }
    [[nodiscard]] inline std::uint64_t n_gates() const { return _gate_list.size(); }
    [[nodiscard]] std::set<std::string> key_set() const;
    [[nodiscard]] inline const GateWithKey& get_gate_at(std::uint64_t idx) const {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error("Circuit::get_gate_at(std::uint64_t): index out of bounds");
        }
        return _gate_list[idx];
    }
    [[nodiscard]] inline std::optional<std::string> get_param_key_at(std::uint64_t idx) const {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error(
                "Circuit::get_parameter_key(std::uint64_t): index out of bounds");
        }
        const auto& gate = _gate_list[idx];
        if (gate.index() == 0) return std::nullopt;
        return std::get<1>(gate).second;
    }

    [[nodiscard]] std::uint64_t calculate_depth() const;

    void add_gate(const Gate<Prec>& gate) { _gate_list.push_back(gate); }
    void add_param_gate(const ParamGate<Prec>& param_gate, std::string_view parameter_key) {
        _gate_list.push_back(std::make_pair(param_gate, std::string(parameter_key)));
    }

    void add_circuit(const Circuit<Prec>& circuit);
    void add_circuit(Circuit<Prec>&& circuit);

    template <ExecutionSpace Space>
    void update_quantum_state(StateVector<Prec, Space>& state,
                              const std::map<std::string, double>& parameters = {},
                              std::uint64_t seed = std::random_device{}()) const;
    template <ExecutionSpace Space>
    void update_quantum_state(StateVector<Prec, Space>& state,
                              ClassicalRegister& classical_register,
                              const std::map<std::string, double>& parameters = {},
                              std::uint64_t seed = std::random_device{}()) const;
    template <ExecutionSpace Space>
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              const std::map<std::string, std::vector<double>>& parameters = {},
                              std::uint64_t seed = std::random_device{}()) const;
    template <ExecutionSpace Space>
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              ClassicalRegisterBatched& classical_register,
                              const std::map<std::string, std::vector<double>>& parameters = {},
                              std::uint64_t seed = std::random_device{}()) const;

    Circuit copy() const;

    Circuit get_inverse() const;

    template <ExecutionSpace Space>
    void optimize(std::uint64_t max_block_size = 3);

    friend void to_json(Json& j, const Circuit& circuit) {
        j = Json{{"gate_list", Json::array()}};
        for (auto&& gate : circuit.gate_list()) {
            if (gate.index() == 0)
                j["gate_list"].emplace_back(Json{{"gate", std::get<0>(gate)}});
            else
                j["gate_list"].emplace_back(
                    Json{{"gate", std::get<1>(gate).first}, {"key", std::get<1>(gate).second}});
        }
    }

    friend void from_json(const Json& j, Circuit& circuit) {
        circuit = Circuit();
        const Json& tmp_list = j.at("gate_list");
        for (const Json& gate_with_key : tmp_list) {
            if (gate_with_key.contains("key")) {
                circuit.add_param_gate(gate_with_key.at("gate").get<ParamGate<Prec>>(),
                                       gate_with_key.at("key").get<std::string>());
            } else {
                circuit.add_gate(gate_with_key.at("gate").get<Gate<Prec>>());
            }
        }
    }

    /**
     * @brief サンプリングされうるすべてのパターンに対して，それぞれが何回選ばれたかを返す
     */
    template <ExecutionSpace Space>
    std::vector<std::pair<StateVector<Prec, Space>, std::int64_t>> simulate_noise(
        const StateVector<Prec, Space>& initial_state,
        std::uint64_t sampling_count,
        const std::map<std::string, double>& parameters = {},
        std::uint64_t seed = 0) const;

    template <ExecutionSpace Space>
    std::unordered_map<std::string, double> compute_expectation_gradient_backprop(
        StateVector<Prec, Space>& state,
        StateVector<Prec, Space>& bistate,
        const std::map<std::string, double>& parameters);

    template <ExecutionSpace Space>
    std::unordered_map<std::string, double> compute_expectation_gradient(
        const Operator<Prec, Space>& observable, const std::map<std::string, double>& parameters);

private:
    std::vector<GateWithKey> _gate_list;

    [[nodiscard]] std::uint64_t required_n_qubits() const;
    void check_state_vector_n_qubits(std::uint64_t n_qubits) const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
using CircuitSingleParameters = std::map<std::string, double>;
using CircuitBatchedParameters = std::map<std::string, std::vector<double>>;
using CircuitParameterVariant =
    std::variant<std::monostate, CircuitBatchedParameters, CircuitSingleParameters>;

template <Precision Prec, ExecutionSpace Space>
void update_circuit_state(const Circuit<Prec>& circuit,
                          StateVector<Prec, Space>* state,
                          ClassicalRegisterVariant classical_register,
                          const CircuitParameterVariant& parameter_arg,
                          std::optional<std::uint64_t> seed,
                          const nb::kwargs& kwargs) {
    CircuitSingleParameters parameters;
    if (const auto* value = std::get_if<CircuitSingleParameters>(&parameter_arg)) {
        parameters = *value;
    } else if (const auto* value = std::get_if<CircuitBatchedParameters>(&parameter_arg)) {
        if (!value->empty()) {
            throw std::runtime_error(
                "Circuit::update_quantum_state(): StateVector requires dict[str, float] "
                "parameters.");
        }
    }
    for (auto&& [key, param] : kwargs) {
        if (!std::holds_alternative<std::monostate>(parameter_arg)) {
            throw std::runtime_error(
                "Circuit::update_quantum_state(): parameters were specified both as dict and "
                "kwargs.");
        }
        parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
    }

    std::visit(
        Overloaded{[&](std::monostate) {
                       circuit.template update_quantum_state<Space>(
                           *state, parameters, seed.value_or(std::random_device{}()));
                   },
                   [&](ClassicalRegister* reg) {
                       circuit.template update_quantum_state<Space>(
                           *state, *reg, parameters, seed.value_or(std::random_device{}()));
                   },
                   [&](ClassicalRegisterBatched*) {
                       throw std::runtime_error(
                           "Circuit::update_quantum_state(): ClassicalRegisterBatched cannot be "
                           "used with StateVector.");
                   }},
        classical_register);
}

template <Precision Prec, ExecutionSpace Space>
void update_circuit_state(const Circuit<Prec>& circuit,
                          StateVectorBatched<Prec, Space>* states,
                          ClassicalRegisterVariant classical_register,
                          const CircuitParameterVariant& parameter_arg,
                          std::optional<std::uint64_t> seed,
                          const nb::kwargs& kwargs) {
    CircuitBatchedParameters parameters;
    if (const auto* value = std::get_if<CircuitBatchedParameters>(&parameter_arg)) {
        parameters = *value;
    } else if (const auto* value = std::get_if<CircuitSingleParameters>(&parameter_arg)) {
        if (!value->empty()) {
            throw std::runtime_error(
                "Circuit::update_quantum_state(): StateVectorBatched requires dict[str, "
                "Sequence[float]] parameters.");
        }
    }
    for (auto&& [key, param] : kwargs) {
        if (!std::holds_alternative<std::monostate>(parameter_arg)) {
            throw std::runtime_error(
                "Circuit::update_quantum_state(): parameters were specified both as dict and "
                "kwargs.");
        }
        parameters[nb::cast<std::string>(key)] = nb::cast<std::vector<double>>(param);
    }

    std::visit(
        Overloaded{[&](std::monostate) {
                       circuit.template update_quantum_state<Space>(
                           *states, parameters, seed.value_or(std::random_device{}()));
                   },
                   [&](ClassicalRegister*) {
                       throw std::runtime_error(
                           "Circuit::update_quantum_state(): ClassicalRegister cannot be used "
                           "with StateVectorBatched.");
                   },
                   [&](ClassicalRegisterBatched* reg) {
                       circuit.template update_quantum_state<Space>(
                           *states, *reg, parameters, seed.value_or(std::random_device{}()));
                   }},
        classical_register);
}

template <Precision Prec>
void register_circuit_update_quantum_state(nb::class_<Circuit<Prec>>& c) {
    using namespace nb::literals;

    constexpr const char* update_signature =
        "def update_quantum_state(self, state: StateVector | StateVectorBatched, "
        "*, params: Mapping[str, float] | Mapping[str, Sequence[float]] | None = None, "
        "classical_register: scaluq.scaluq_core.ClassicalRegister | "
        "scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = "
        "None, **kwargs) -> None";
    auto update_doc_str =
        DocString()
            .desc("Apply circuit to `state`. `state` in args is directly updated.")
            .desc(
                "Parameters can be passed as a dict or keyword arguments. For batched states, "
                "parameter values must be sequences with length equal to batch size.")
            .arg("state", "StateVector | StateVectorBatched", "State vector to be updated.")
            .arg("params",
                 "Mapping[str, float] | Mapping[str, Sequence[float]] | None",
                 "Parameter dictionary. Use float values for `StateVector` and sequences for "
                 "`StateVectorBatched`.")
            .arg("classical_register",
                 "ClassicalRegister | ClassicalRegisterBatched | None",
                 "Classical register to be used by gates in the circuit.")
            .arg("seed", "int | None", "Seed for random number generator.")
            .build_as_google_style();

    c.def(
        "update_quantum_state",
        [](const Circuit<Prec>& circuit,
           GateStateVariant<Prec> state,
           CircuitParameterVariant parameters,
           ClassicalRegisterVariant classical_register,
           std::optional<std::uint64_t> seed,
           nb::kwargs kwargs) {
            std::visit(
                [&](auto* state_ptr) {
                    update_circuit_state<Prec>(
                        circuit, state_ptr, classical_register, parameters, seed, kwargs);
                },
                state);
        },
        "state"_a,
        nb::kw_only(),
        "params"_a = std::monostate{},
        "classical_register"_a = std::monostate{},
        "seed"_a = std::nullopt,
        "kwargs"_a,
        nb::sig(update_signature),
        update_doc_str.c_str());
}

template <Precision Prec, ExecutionSpace Space>
void register_circuit_space_bindings(nb::class_<Circuit<Prec>>& c) {
    using namespace nb::literals;

    c.def("optimize",
          nb::overload_cast<std::uint64_t>(&Circuit<Prec>::template optimize<Space>),
          "max_block_size"_a = 3,
          "Optimize circuit. Create qubit dependency tree and merge neighboring gates if the "
          "new gate has less than or equal to `max_block_size` or the new gate is Pauli.")
        .def(
            "simulate_noise",
            [](const Circuit<Prec>& circuit,
               const StateVector<Prec, Space>& initial_state,
               std::uint64_t sampling_count,
               const std::map<std::string, double>& parameters,
               std::optional<std::uint64_t> seed) {
                return circuit.template simulate_noise<Space>(
                    initial_state,
                    sampling_count,
                    parameters,
                    seed.value_or(std::random_device{}()));
            },
            "initial_state"_a,
            "sampling_count"_a,
            "parameters"_a = std::map<std::string, double>{},
            "seed"_a = std::nullopt,
            "Simulate noise circuit. Return all the possible states and their counts.")
        .def("compute_expectation_gradient_backprop",
             &Circuit<Prec>::template compute_expectation_gradient_backprop<Space>,
             "state"_a,
             "bistate"_a,
             "parameters"_a,
             "Low-level implementation for expectation gradient that assumes the forward state and "
             "observable-applied bistate are already prepared, and computes gradient using back "
             "propagation.")
        .def("compute_expectation_gradient",
             &Circuit<Prec>::template compute_expectation_gradient<Space>,
             "observable"_a,
             "parameters"_a,
             "Compute gradient of expectation value of observable using back propagation.");
}

template <Precision Prec>
void bind_circuit_circuit_hpp(nb::module_& m) {
    using namespace nb::literals;

    auto c = nb::class_<Circuit<Prec>>(
        m,
        "Circuit",
        DocString()
            .desc("Quantum circuit representation.")
            .ex(DocString::Code(
                {">>> circuit = Circuit()", ">>> print(circuit.to_json())", "{\"gate_list\":[]}"}))
            .build_as_google_style()
            .c_str());

    c.def(nb::init<>(), "Initialize empty circuit.")
        .def("gate_list",
             &Circuit<Prec>::gate_list,
             "Get property of `gate_list`.",
             nb::rv_policy::reference)
        .def("n_gates", &Circuit<Prec>::n_gates, "Get property of `n_gates`.")
        .def("key_set", &Circuit<Prec>::key_set, "Get set of keys of parameters.")
        .def("get_gate_at", &Circuit<Prec>::get_gate_at, "index"_a, "Get reference of i-th gate.")
        .def("get_param_key_at",
             &Circuit<Prec>::get_param_key_at,
             "index"_a,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit<Prec>::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate<Prec>&>(&Circuit<Prec>::add_gate),
             "gate"_a,
             "Add gate. Given gate is copied.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view>(
                 &Circuit<Prec>::add_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "Add parametric gate with specifying key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit<Prec>&>(&Circuit<Prec>::add_circuit),
             "other"_a,
             "Add all gates in specified circuit. Given gates are copied.")
        .def("copy",
             &Circuit<Prec>::copy,
             "Copy circuit. Returns a new circuit instance with all gates copied by reference.")
        .def("get_inverse",
             &Circuit<Prec>::get_inverse,
             "Get inverse of circuit. All the gates are newly created.")
        .def(
            "to_json",
            [](const Circuit<Prec>& circuit) { return Json(circuit).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](Circuit<Prec>& circuit, const std::string& str) {
                circuit = nlohmann::json::parse(str);
            },
            "json_str"_a,
            "Read an object from the JSON representation of the circuit.");

    register_circuit_update_quantum_state<Prec>(c);
    register_circuit_space_bindings<Prec, ExecutionSpace::Host>(c);
    register_circuit_space_bindings<Prec, ExecutionSpace::HostSerial>(c);
#ifdef SCALUQ_USE_CUDA
    register_circuit_space_bindings<Prec, ExecutionSpace::Default>(c);
#endif  // SCALUQ_USE_CUDA
}
}  // namespace internal
#endif
}  // namespace scaluq
