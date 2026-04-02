#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../types.hpp"

namespace scaluq {

class ClassicalRegister {
private:
    const std::vector<bool>* _reg = nullptr;
    std::uint64_t _offset = 0;
    std::uint64_t _size = 0;

public:
    explicit ClassicalRegister(const std::vector<bool>& reg) : _reg(&reg), _size(reg.size()) {}
    ClassicalRegister(const std::vector<bool>& reg, std::uint64_t offset, std::uint64_t size)
        : _reg(&reg), _offset(offset), _size(size) {
        if (_offset + _size > reg.size()) {
            throw std::runtime_error("ClassicalRegister: offset and size are out of range.");
        }
    }

    [[nodiscard]] bool operator[](std::uint64_t index) const {
        if (index >= _size) return false;
        return (*_reg)[_offset + index];
    }
    [[nodiscard]] std::uint64_t size() const { return _size; }
    [[nodiscard]] bool empty() const { return _size == 0; }
};

/**
 * @brief Classical condition evaluated against the circuit register.
 *
 * Any callable convertible to `std::function<bool(const ClassicalRegister&)>`
 * can be used here, such as:
 * - lambda expressions
 * - function pointers
 * - functors
 * - `std::function`
 *
 * @example
 * `[](const ClassicalRegister& reg) {`
 * `    return reg.size() > 1 && reg[0] && reg[1];`
 * `}`
 */
using ClassicalCondition = std::function<bool(const ClassicalRegister&)>;

template <Precision Prec>
class Circuit {
public:
    using GateWithKey = std::variant<Gate<Prec>, std::pair<ParamGate<Prec>, std::string>>;
    explicit Circuit(std::uint64_t n_classical_bits = 0)
        : _reg(n_classical_bits, false), _n_classical_bits(n_classical_bits) {}

    [[nodiscard]] inline const std::vector<GateWithKey>& gate_list() const { return _gate_list; }
    [[nodiscard]] inline std::uint64_t n_gates() const { return _gate_list.size(); }
    [[nodiscard]] std::set<std::string> key_set() const;
    [[nodiscard]] inline const std::vector<bool>& classical_register() const { return _reg; }
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
    [[nodiscard]] inline std::optional<ClassicalCondition> get_classical_condition_at(
        std::uint64_t idx) const {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error(
                "Circuit::get_classical_condition_at(std::uint64_t): index out of bounds");
        }
        return _classical_conditions[idx];
    }

    [[nodiscard]] std::uint64_t calculate_depth() const;

    void add_gate(const Gate<Prec>& gate) {
        if (gate.gate_type() == GateType::Measurement &&
            MeasurementGate<Prec>(gate)->classical_bit_index() >= _n_classical_bits) {
            throw std::runtime_error(
                "Circuit::add_gate: measurement classical bit index is out of range.");
        }
        _gate_list.push_back(gate);
        _classical_conditions.push_back(std::nullopt);
    }
    void add_param_gate(const ParamGate<Prec>& param_gate, std::string_view parameter_key) {
        _gate_list.push_back(std::make_pair(param_gate, std::string(parameter_key)));
        _classical_conditions.push_back(std::nullopt);
    }

    /**
     * @brief Add a gate guarded by a user-defined classical condition.
     *
     * The condition is evaluated as `condition(const ClassicalRegister&)`.
     * Missing register bits should be handled by the callable itself.
     *
     * @example
     * `circuit.add_conditional_gate(gate::X<Prec>(2), [](const ClassicalRegister& reg) {`
     * `    return reg[0] ^ reg[1];`
     * `});`
     */
    void add_conditional_gate(const Gate<Prec>& gate, ClassicalCondition condition) {
        add_gate(gate);
        _classical_conditions.back() = std::move(condition);
    }
    void add_conditional_gate(const Gate<Prec>& gate,
                              std::uint64_t classical_bit_index,
                              bool expected_value) {
        if (classical_bit_index >= _n_classical_bits) {
            throw std::runtime_error(
                "Circuit::add_conditional_gate: classical bit index is out of range.");
        }
        add_conditional_gate(gate,
                             [classical_bit_index, expected_value](const ClassicalRegister& reg) {
                                 return reg[classical_bit_index] == expected_value;
                             });
    }

    /**
     * @brief Add a parametric gate guarded by a user-defined classical condition.
     *
     * The condition is evaluated as `condition(const ClassicalRegister&)`.
     * Missing register bits should be handled by the callable itself.
     *
     * @example
     * `circuit.add_conditional_param_gate(gate::ParamRX<Prec>(0), "theta",`
     * `    [](const ClassicalRegister& reg) {`
     * `        return reg[0] ^ reg[1];`
     * `    });`
     */
    void add_conditional_param_gate(const ParamGate<Prec>& param_gate,
                                    std::string_view parameter_key,
                                    ClassicalCondition condition) {
        add_param_gate(param_gate, parameter_key);
        _classical_conditions.back() = std::move(condition);
    }
    void add_conditional_param_gate(const ParamGate<Prec>& param_gate,
                                    std::string_view parameter_key,
                                    std::uint64_t classical_bit_index,
                                    bool expected_value) {
        if (classical_bit_index >= _n_classical_bits) {
            throw std::runtime_error(
                "Circuit::add_conditional_param_gate: classical bit index is out of range.");
        }
        add_conditional_param_gate(
            param_gate,
            parameter_key,
            [classical_bit_index, expected_value](const ClassicalRegister& reg) {
                return reg[classical_bit_index] == expected_value;
            });
    }

    void add_circuit(const Circuit<Prec>& circuit);
    void add_circuit(Circuit<Prec>&& circuit);

    template <ExecutionSpace Space>
    void update_quantum_state(StateVector<Prec, Space>& state,
                              const std::map<std::string, double>& parameters = {},
                              std::uint64_t seed = std::random_device{}());
    template <ExecutionSpace Space>
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              const std::map<std::string, std::vector<double>>& parameters = {},
                              std::uint64_t seed = std::random_device{}());

    Circuit copy() const;

    Circuit get_inverse() const;

    template <ExecutionSpace Space>
    void optimize(std::uint64_t max_block_size = 3);

    friend void to_json(Json& j, const Circuit& circuit) {
        j = Json{{"gate_list", Json::array()}, {"n_classical_bits", circuit._n_classical_bits}};
        for (std::uint64_t idx = 0; idx < circuit.gate_list().size(); ++idx) {
            if (circuit._classical_conditions[idx].has_value()) {
                throw std::runtime_error(
                    "Circuit::to_json: classically controlled instructions cannot be serialized.");
            }
            const auto& gate = circuit.gate_list()[idx];
            Json gate_json = gate.index() == 0 ? Json{{"gate", std::get<0>(gate)}}
                                               : Json{{"gate", std::get<1>(gate).first},
                                                      {"key", std::get<1>(gate).second}};
            j["gate_list"].emplace_back(std::move(gate_json));
        }
    }

    friend void from_json(const Json& j, Circuit& circuit) {
        circuit = Circuit(j.value("n_classical_bits", std::uint64_t{0}));
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

private:
    std::vector<GateWithKey> _gate_list;
    std::vector<std::optional<ClassicalCondition>> _classical_conditions;
    std::vector<bool> _reg;
    std::uint64_t _n_classical_bits = 0;

    [[nodiscard]] bool has_classical_conditions() const {
        return std::any_of(_classical_conditions.begin(),
                           _classical_conditions.end(),
                           [](const auto& condition) { return condition.has_value(); });
    }
    [[nodiscard]] std::uint64_t required_n_qubits() const;
    void check_state_vector_n_qubits(std::uint64_t n_qubits) const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space, typename ClassT>
void bind_circuit_execution_methods(ClassT& circuit_def) {
    circuit_def
        .def(
            "update_quantum_state",
            [](Circuit<Prec>& circuit,
               StateVector<Prec, Space>& state,
               const std::map<std::string, double>& parameters,
               std::optional<std::uint64_t> seed) {
                circuit.template update_quantum_state<Space>(
                    state, parameters, seed.value_or(std::random_device{}()));
            },
            "state"_a,
            "params"_a = std::map<std::string, double>{},
            "seed"_a = std::nullopt,
            "Execute the circuit, including measurement and classical branching.")
        .def(
            "update_quantum_state",
            [](Circuit<Prec>& circuit, StateVector<Prec, Space>& state, nb::kwargs kwargs) {
                std::map<std::string, double> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
                }
                circuit.template update_quantum_state<Space>(state, parameters);
            },
            "state"_a,
            "kwargs"_a,
            "Execute the circuit. If the circuit contains parametric gates, pass parameters as "
            "\"name=value\" in kwargs.")
        .def(
            "update_quantum_state",
            [](Circuit<Prec>& circuit,
               StateVectorBatched<Prec, Space>& states,
               const std::map<std::string, std::vector<double>>& parameters,
               std::optional<std::uint64_t> seed) {
                circuit.template update_quantum_state<Space>(
                    states, parameters, seed.value_or(std::random_device{}()));
            },
            "states"_a,
            "params"_a = std::map<std::string, std::vector<double>>{},
            "seed"_a = std::nullopt,
            "Execute the circuit on a batched state vector.")
        .def(
            "update_quantum_state",
            [](Circuit<Prec>& circuit, StateVectorBatched<Prec, Space>& states, nb::kwargs kwargs) {
                std::map<std::string, std::vector<double>> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<std::vector<double>>(param);
                }
                circuit.template update_quantum_state<Space>(states, parameters);
            },
            "states"_a,
            "kwargs"_a,
            "Execute the circuit on a batched state vector. If the circuit contains parametric "
            "gates, pass parameters as \"name=[value1, value2, ...]\" in kwargs.")
        .def("optimize",
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
            "Simulate noise circuit. Return all the possible states and their counts.");
}

template <Precision Prec>
void bind_circuit_circuit_hpp(nb::module_& m) {
    auto circuit_def = nb::class_<Circuit<Prec>>(
        m,
        "Circuit",
        DocString()
            .desc("Quantum circuit representation.")
            .ex(DocString::Code({">>> circuit = Circuit()",
                                 ">>> print(circuit.to_json())",
                                 "{\"gate_list\":[],\"n_classical_bits\":0}"}))
            .build_as_google_style()
            .c_str());

    circuit_def.def(nb::init<>(), "Initialize circuit without classical registers.")
        .def(nb::init<std::uint64_t>(),
             "n_classical_bits"_a,
             "Initialize circuit with `n_classical_bits` classical registers.")
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
        .def("classical_register",
             &Circuit<Prec>::classical_register,
             "Get property of `classical_register`.")
        .def("get_classical_condition_at",
             &Circuit<Prec>::get_classical_condition_at,
             "index"_a,
             "Get classical condition of i-th gate. If it is not conditional, return None.")
        .def("calculate_depth", &Circuit<Prec>::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate<Prec>&>(&Circuit<Prec>::add_gate),
             "gate"_a,
             "Add gate. Given gate is copied.")
        .def("add_conditional_gate",
             nb::overload_cast<const Gate<Prec>&, ClassicalCondition>(
                 &Circuit<Prec>::add_conditional_gate),
             "gate"_a,
             "condition"_a,
             "Add gate with a user-defined classical condition.")
        .def("add_conditional_gate",
             nb::overload_cast<const Gate<Prec>&, std::uint64_t, bool>(
                 &Circuit<Prec>::add_conditional_gate),
             "gate"_a,
             "classical_bit_index"_a,
             "expected_value"_a,
             "Add gate with classical condition.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view>(
                 &Circuit<Prec>::add_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "Add parametric gate with specifying key. Given param_gate is copied.")
        .def("add_conditional_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view, ClassicalCondition>(
                 &Circuit<Prec>::add_conditional_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "condition"_a,
             "Add parametric gate with a user-defined classical condition.")
        .def("add_conditional_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view, std::uint64_t, bool>(
                 &Circuit<Prec>::add_conditional_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "classical_bit_index"_a,
             "expected_value"_a,
             "Add parametric gate with classical condition.")
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
    bind_circuit_execution_methods<Prec, ExecutionSpace::Host>(circuit_def);
    bind_circuit_execution_methods<Prec, ExecutionSpace::HostSerial>(circuit_def);
#ifdef SCALUQ_USE_CUDA
    bind_circuit_execution_methods<Prec, ExecutionSpace::Default>(circuit_def);
#endif  // SCALUQ_USE_CUDA
}
}  // namespace internal
#endif
}  // namespace scaluq
