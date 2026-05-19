#pragma once

#include <algorithm>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

#include "../classical_register/classical_register.hpp"
#include "../classical_register/classical_register_batched.hpp"
#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../operator/operator.hpp"
#include "../types.hpp"

namespace scaluq {

using ClassicalCondition = std::function<bool(const ClassicalRegister&)>;

template <Precision Prec>
class Circuit;

template <Precision Prec>
struct CircuitStorage;

template <Precision Prec>
using CircuitInstruction = std::variant<Gate<Prec>, ParamGateWithTag, Circuit<Prec>>;

template <Precision Prec>
[[nodiscard]] bool is_gate(const CircuitInstruction<Prec>& instruction) {
    return instruction.index() == 0;
}
template <Precision Prec>
[[nodiscard]] bool is_param_gate(const CircuitInstruction<Prec>& instruction) {
    return instruction.index() == 1;
}
template <Precision Prec>
[[nodiscard]] bool is_circuit(const CircuitInstruction<Prec>& instruction) {
    return instruction.index() == 2;
}
template <Precision Prec>
[[nodiscard]] const Gate<Prec>& get_gate(const CircuitInstruction<Prec>& instruction) {
    return std::get<0>(instruction);
}
template <Precision Prec>
[[nodiscard]] const std::pair<ParamGate<Prec>, std::string>& get_param_gate_with_key(
    const CircuitInstruction<Prec>& instruction) {
    return std::get<1>(instruction);
}
template <Precision Prec>
[[nodiscard]] const Circuit<Prec>& get_circuit(const CircuitInstruction<Prec>& instruction) {
    return std::get<2>(instruction);
}

template <Precision Prec>
class Circuit {
public:
    using GateWithKey = std::variant<Gate<Prec>, std::pair<ParamGate<Prec>, std::string>>;
    using Instruction = CircuitInstruction<Prec>;
    Circuit();

    [[nodiscard]] inline const std::vector<Instruction>& instructions() const {
        return _storage->instructions;
    }
    [[nodiscard]] inline std::uint64_t n_instructions() const {
        return _storage->instructions.size();
    }
    [[nodiscard]] std::set<std::string> key_set() const;
    [[nodiscard]] inline const Instruction& get_instruction_at(std::uint64_t idx) const {
        if (idx >= _storage->instructions.size()) {
            throw std::runtime_error(
                "Circuit::get_instruction_at(std::uint64_t): index out of bounds");
        }
        return _storage->instructions[idx];
    }
    [[nodiscard]] inline std::optional<std::string> get_param_key_at(std::uint64_t idx) const {
        if (idx >= _storage->instructions.size()) {
            throw std::runtime_error(
                "Circuit::get_parameter_key(std::uint64_t): index out of bounds");
        }
        const auto& instruction = _storage->instructions[idx];
        if (!is_param_gate(instruction)) return std::nullopt;
        return get_param_gate_with_key(instruction).second;
    }
    [[nodiscard]] inline std::optional<ClassicalCondition> get_classical_condition_at(
        std::uint64_t idx) const {
        if (idx >= _storage->instructions.size()) {
            throw std::runtime_error(
                "Circuit::get_classical_condition_at(std::uint64_t): index out of bounds");
        }
        return _storage->classical_conditions[idx];
    }

    [[nodiscard]] std::uint64_t calculate_depth() const;

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
    [[nodiscard]] Circuit optimize(std::uint64_t max_block_size = 3) const;

    friend void to_json(Json& j, const Circuit& circuit) {
        j = Json{{"gate_list", Json::array()}};
        for (std::uint64_t idx = 0; idx < circuit.instructions().size(); ++idx) {
            if (circuit._storage->classical_conditions[idx].has_value()) {
                throw std::runtime_error(
                    "Circuit::to_json: classically controlled instructions cannot be serialized.");
            }
            const auto& instruction = circuit.instructions()[idx];
            if (is_gate(instruction)) {
                j["gate_list"].emplace_back(Json{{"gate", get_gate(instruction)}});
            } else if (is_param_gate(instruction)) {
                const auto& [param_gate, key] = get_param_gate_with_key(instruction);
                j["gate_list"].emplace_back(Json{{"gate", param_gate}, {"key", key}});
            } else {
                j["gate_list"].emplace_back(Json{{"circuit", get_circuit(instruction)}});
            }
        }
    }

    friend void from_json(const Json& j, Circuit& circuit) {
        auto storage = std::make_shared<CircuitStorage<Prec>>();
        const Json& tmp_list = j.at("gate_list");
        for (const Json& instruction : tmp_list) {
            if (instruction.contains("circuit")) {
                storage->instructions.push_back(instruction.at("circuit").get<Circuit<Prec>>());
            } else if (instruction.contains("key")) {
                storage->instructions.push_back(
                    std::make_pair(instruction.at("gate").get<ParamGate<Prec>>(),
                                   instruction.at("key").get<std::string>()));
            } else {
                storage->instructions.push_back(instruction.at("gate").get<Gate<Prec>>());
            }
            storage->classical_conditions.push_back(std::nullopt);
        }
        circuit = Circuit(std::move(storage));
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
    template <Precision>
    friend class CircuitBuilder;

    explicit Circuit(std::shared_ptr<const CircuitStorage<Prec>> storage)
        : _storage(std::move(storage)) {}

    std::shared_ptr<const CircuitStorage<Prec>> _storage;

    [[nodiscard]] std::uint64_t required_operand_qubit_mask() const;
    [[nodiscard]] std::uint64_t required_n_qubits() const;
    [[nodiscard]] bool has_classical_instructions() const;
    void check_state_vector_n_qubits(std::uint64_t n_qubits) const;

    template <ExecutionSpace Space>
    void update_quantum_state_internal(internal::ExecutionContext<Prec, Space> context,
                                       const std::map<std::string, double>& parameters) const;
    template <ExecutionSpace Space>
    void update_quantum_state_internal(
        internal::ExecutionContextBatched<Prec, Space> context,
        const std::map<std::string, std::vector<double>>& parameters) const;
};

template <Precision Prec>
struct CircuitStorage {
    std::vector<CircuitInstruction<Prec>> instructions;
    std::vector<std::optional<ClassicalCondition>> classical_conditions;
};

template <Precision Prec>
Circuit<Prec>::Circuit() : _storage(std::make_shared<CircuitStorage<Prec>>()) {}

template <Precision Prec>
class CircuitBuilder {
public:
    using Instruction = typename Circuit<Prec>::Instruction;

    CircuitBuilder() = default;

    [[nodiscard]] std::uint64_t n_instructions() const { return _instructions.size(); }

    void add_gate(const Gate<Prec>& gate) {
        _instructions.push_back(gate);
        _classical_conditions.push_back(std::nullopt);
    }
    void add_param_gate(const ParamGate<Prec>& param_gate, std::string_view parameter_key) {
        _instructions.push_back(std::make_pair(param_gate, std::string(parameter_key)));
        _classical_conditions.push_back(std::nullopt);
    }
    void add_conditional_gate(const Gate<Prec>& gate, ClassicalCondition condition) {
        _instructions.push_back(gate);
        _classical_conditions.push_back(std::move(condition));
    }
    void add_conditional_gate(const Gate<Prec>& gate,
                              std::uint64_t classical_bit_index,
                              bool expected_value) {
        add_conditional_gate(gate,
                             [classical_bit_index, expected_value](const ClassicalRegister& reg) {
                                 return reg[classical_bit_index] == expected_value;
                             });
    }
    void add_conditional_param_gate(const ParamGate<Prec>& param_gate,
                                    std::string_view parameter_key,
                                    ClassicalCondition condition) {
        _instructions.push_back(std::make_pair(param_gate, std::string(parameter_key)));
        _classical_conditions.push_back(std::move(condition));
    }
    void add_conditional_param_gate(const ParamGate<Prec>& param_gate,
                                    std::string_view parameter_key,
                                    std::uint64_t classical_bit_index,
                                    bool expected_value) {
        add_conditional_param_gate(
            param_gate,
            parameter_key,
            [classical_bit_index, expected_value](const ClassicalRegister& reg) {
                return reg[classical_bit_index] == expected_value;
            });
    }
    void extend_circuit(const Circuit<Prec>& circuit) {
        _instructions.insert(_instructions.end(),
                             circuit._storage->instructions.begin(),
                             circuit._storage->instructions.end());
        _classical_conditions.insert(_classical_conditions.end(),
                                     circuit._storage->classical_conditions.begin(),
                                     circuit._storage->classical_conditions.end());
    }
    void add_circuit(const Circuit<Prec>& circuit) {
        _instructions.push_back(circuit);
        _classical_conditions.push_back(std::nullopt);
    }
    void add_conditional_circuit(const Circuit<Prec>& circuit, ClassicalCondition condition) {
        _instructions.push_back(circuit);
        _classical_conditions.push_back(std::move(condition));
    }
    void add_conditional_circuit(const Circuit<Prec>& circuit,
                                 std::uint64_t classical_bit_index,
                                 bool expected_value) {
        add_conditional_circuit(
            circuit, [classical_bit_index, expected_value](const ClassicalRegister& reg) {
                return reg[classical_bit_index] == expected_value;
            });
    }

    [[nodiscard]] Circuit<Prec> build() const {
        auto storage = std::make_shared<CircuitStorage<Prec>>();
        storage->instructions = _instructions;
        storage->classical_conditions = _classical_conditions;
        return Circuit<Prec>(std::move(storage));
    }

private:
    std::vector<Instruction> _instructions;
    std::vector<std::optional<ClassicalCondition>> _classical_conditions;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void register_circuit_space_bindings(nb::class_<Circuit<Prec>>& c) {
    using namespace nb::literals;

    c.def(
         "update_quantum_state",
         [&](const Circuit<Prec>& circuit, StateVector<Prec, Space>& state, nb::kwargs kwargs) {
             std::map<std::string, double> parameters;
             for (auto&& [key, param] : kwargs) {
                 parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
             }
             circuit.update_quantum_state(state, parameters);
         },
         "state"_a,
         "kwargs"_a,
         "Apply gate to the StateVector. StateVector in args is directly updated. If the "
         "circuit contains parametric gate, you have to give real value of parameter as "
         "\"name=value\" format in kwargs.")
        .def(
            "update_quantum_state",
            [](const Circuit<Prec>& circuit,
               StateVector<Prec, Space>& state,
               const std::map<std::string, double>& parameters,
               std::optional<std::uint64_t> seed) {
                circuit.update_quantum_state(
                    state, parameters, seed.value_or(std::random_device{}()));
            },
            "state"_a,
            "params"_a = std::map<std::string, double>{},
            "seed"_a = std::nullopt,
            "Apply gate to the StateVector. StateVector in args is directly updated. If the "
            "circuit contains parametric gate, you have to give real value of parameter as "
            "dict[str, float] in 2nd arg.")
        .def(
            "update_quantum_state",
            [](const Circuit<Prec>& circuit,
               StateVector<Prec, Space>& state,
               ClassicalRegister& classical_register,
               const std::map<std::string, double>& parameters,
               std::optional<std::uint64_t> seed) {
                circuit.update_quantum_state(
                    state, classical_register, parameters, seed.value_or(std::random_device{}()));
            },
            "state"_a,
            "classical_register"_a,
            "params"_a = std::map<std::string, double>{},
            "seed"_a = std::nullopt,
            "Apply gate to the StateVector with classical register and optional seed.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Prec>& circuit,
                StateVector<Prec, Space>& state,
                ClassicalRegister& classical_register,
                nb::kwargs kwargs) {
                std::map<std::string, double> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
                }
                circuit.update_quantum_state(state, classical_register, parameters);
            },
            "state"_a,
            "classical_register"_a,
            "kwargs"_a,
            "Apply gate to the StateVector with classical register. If the circuit contains "
            "parametric gate, give parameter values as \"name=value\" in kwargs.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Prec>& circuit,
                StateVectorBatched<Prec, Space>& states,
                nb::kwargs kwargs) {
                std::map<std::string, std::vector<double>> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<std::vector<double>>(param);
                }
                circuit.update_quantum_state(states, parameters);
            },
            "state"_a,
            "kwargs"_a,
            "Apply gate to the StateVectorBatched. StateVectorBatched in args is directly updated. "
            "If the circuit contains parametric gate, you have to give real value of parameter as "
            "\"name=[value1, value2, ...]\" format in kwargs.")
        .def(
            "update_quantum_state",
            [](const Circuit<Prec>& circuit,
               StateVectorBatched<Prec, Space>& states,
               const std::map<std::string, std::vector<double>>& parameters,
               std::optional<std::uint64_t> seed) {
                circuit.update_quantum_state(
                    states, parameters, seed.value_or(std::random_device{}()));
            },
            "state"_a,
            "params"_a = std::map<std::string, std::vector<double>>{},
            "seed"_a = std::nullopt,
            "Apply gate to the StateVectorBatched. StateVectorBatched in args is directly updated. "
            "If the circuit contains parametric gate, you have to give real value of parameter as "
            "dict[str, list[float]] in 2nd arg.")
        .def(
            "update_quantum_state",
            [](const Circuit<Prec>& circuit,
               StateVectorBatched<Prec, Space>& states,
               ClassicalRegisterBatched& classical_register,
               const std::map<std::string, std::vector<double>>& parameters,
               std::optional<std::uint64_t> seed) {
                circuit.update_quantum_state(
                    states, classical_register, parameters, seed.value_or(std::random_device{}()));
            },
            "state"_a,
            "classical_register"_a,
            "params"_a = std::map<std::string, std::vector<double>>{},
            "seed"_a = std::nullopt,
            "Apply gate to the StateVectorBatched with classical register and optional seed.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Prec>& circuit,
                StateVectorBatched<Prec, Space>& states,
                ClassicalRegisterBatched& classical_register,
                nb::kwargs kwargs) {
                std::map<std::string, std::vector<double>> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<std::vector<double>>(param);
                }
                circuit.update_quantum_state(states, classical_register, parameters);
            },
            "state"_a,
            "classical_register"_a,
            "kwargs"_a,
            "Apply gate to the StateVectorBatched with classical register. If the circuit "
            "contains parametric gate, give parameter values as "
            "\"name=[value1, value2, ...]\" in kwargs.")
        .def(
            "optimize",
            [](const Circuit<Prec>& circuit, std::uint64_t max_block_size) {
                return circuit.template optimize<Space>(max_block_size);
            },
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
        .def("instructions",
             &Circuit<Prec>::instructions,
             "Get property of `instructions`.",
             nb::rv_policy::reference)
        .def("n_instructions", &Circuit<Prec>::n_instructions, "Get property of `n_instructions`.")
        .def("key_set", &Circuit<Prec>::key_set, "Get set of keys of parameters.")
        .def("get_instruction_at",
             &Circuit<Prec>::get_instruction_at,
             "index"_a,
             "Get reference of i-th instruction.")
        .def("get_param_key_at",
             &Circuit<Prec>::get_param_key_at,
             "index"_a,
             "Get parameter key of i-th instruction. If it is not parametric, return None.")
        .def("get_classical_condition_at",
             &Circuit<Prec>::get_classical_condition_at,
             "index"_a,
             "Get classical condition of i-th instruction. If it is not conditional, return None.")
        .def("calculate_depth", &Circuit<Prec>::calculate_depth, "Get depth of circuit.")
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
        .def_static(
            "from_json",
            [](const std::string& str) {
                return nlohmann::json::parse(str).template get<Circuit<Prec>>();
            },
            "json_str"_a,
            "Create an object from the JSON representation of the circuit.");

    nb::class_<CircuitBuilder<Prec>>(m, "CircuitBuilder", "Mutable builder for Circuit.")
        .def(nb::init<>(), "Initialize empty circuit builder.")
        .def("n_instructions",
             &CircuitBuilder<Prec>::n_instructions,
             "Get property of `n_instructions`.")
        .def("add_gate",
             nb::overload_cast<const Gate<Prec>&>(&CircuitBuilder<Prec>::add_gate),
             "gate"_a,
             "Add gate. Given gate is copied.")
        .def("add_conditional_gate",
             nb::overload_cast<const Gate<Prec>&, ClassicalCondition>(
                 &CircuitBuilder<Prec>::add_conditional_gate),
             "gate"_a,
             "condition"_a,
             "Add gate with a user-defined classical condition.")
        .def("add_conditional_gate",
             nb::overload_cast<const Gate<Prec>&, std::uint64_t, bool>(
                 &CircuitBuilder<Prec>::add_conditional_gate),
             "gate"_a,
             "classical_bit_index"_a,
             "expected_value"_a,
             "Add gate with a condition on a classical bit.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view>(
                 &CircuitBuilder<Prec>::add_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "Add parametric gate with specifying key. Given param_gate is copied.")
        .def("add_conditional_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view, ClassicalCondition>(
                 &CircuitBuilder<Prec>::add_conditional_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "condition"_a,
             "Add parametric gate with a user-defined classical condition.")
        .def("add_conditional_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view, std::uint64_t, bool>(
                 &CircuitBuilder<Prec>::add_conditional_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "classical_bit_index"_a,
             "expected_value"_a,
             "Add parametric gate with a condition on a classical bit.")
        .def("extend_circuit",
             nb::overload_cast<const Circuit<Prec>&>(&CircuitBuilder<Prec>::extend_circuit),
             "other"_a,
             "Add all gates in specified circuit. Given gates are copied.")
        .def("add_circuit",
             &CircuitBuilder<Prec>::add_circuit,
             "other"_a,
             "Add specified circuit as a nested subcircuit.")
        .def("add_conditional_circuit",
             nb::overload_cast<const Circuit<Prec>&, ClassicalCondition>(
                 &CircuitBuilder<Prec>::add_conditional_circuit),
             "other"_a,
             "condition"_a,
             "Add subcircuit with a user-defined classical condition.")
        .def("add_conditional_circuit",
             nb::overload_cast<const Circuit<Prec>&, std::uint64_t, bool>(
                 &CircuitBuilder<Prec>::add_conditional_circuit),
             "other"_a,
             "classical_bit_index"_a,
             "expected_value"_a,
             "Add subcircuit with a condition on a classical bit.")
        .def("build", &CircuitBuilder<Prec>::build, "Build an immutable Circuit.");

    register_circuit_space_bindings<Prec, ExecutionSpace::Host>(c);
    register_circuit_space_bindings<Prec, ExecutionSpace::HostSerial>(c);
#ifdef SCALUQ_USE_CUDA
    register_circuit_space_bindings<Prec, ExecutionSpace::Default>(c);
#endif  // SCALUQ_USE_CUDA
}
}  // namespace internal
#endif
}  // namespace scaluq
