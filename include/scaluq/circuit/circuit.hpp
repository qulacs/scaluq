#pragma once

#include <set>
#include <variant>

#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../types.hpp"

namespace scaluq {

template <std::floating_point Fp, ExecutionSpace Sp>
class Circuit {
public:
    using GateWithKey = std::variant<Gate<Fp, Sp>, std::pair<ParamGate<Fp, Sp>, std::string>>;
    Circuit() = default;
    explicit Circuit(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline std::uint64_t n_qubits() const { return _n_qubits; }
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

    void add_gate(const Gate<Fp, Sp>& gate) {
        check_gate_is_valid(gate);
        _gate_list.push_back(gate);
    }
    void add_gate(Gate<Fp, Sp>&& gate) {
        check_gate_is_valid(gate);
        _gate_list.push_back(std::move(gate));
    }
    void add_param_gate(const ParamGate<Fp, Sp>& param_gate, std::string_view parameter_key) {
        check_gate_is_valid(param_gate);
        _gate_list.push_back(std::make_pair(param_gate, std::string(parameter_key)));
    }
    void add_param_gate(ParamGate<Fp, Sp>&& param_gate, std::string_view parameter_key) {
        check_gate_is_valid(param_gate);
        _gate_list.push_back(std::make_pair(std::move(param_gate), std::string(parameter_key)));
    }

    void add_circuit(const Circuit<Fp, Sp>& circuit);
    void add_circuit(Circuit<Fp, Sp>&& circuit);

    void update_quantum_state(StateVector<Fp, Sp>& state,
                              const std::map<std::string, Fp>& parameters = {}) const;

    Circuit copy() const;

    Circuit get_inverse() const;

    friend void to_json(Json& j, const Circuit& circuit) {
        j = Json{{"n_qubits", circuit.n_qubits()}, {"gate_list", Json::array()}};
        for (auto&& gate : circuit.gate_list()) {
            if (gate.index() == 0)
                j["gate_list"].emplace_back(Json{{"gate", std::get<0>(gate)}});
            else
                j["gate_list"].emplace_back(
                    Json{{"gate", std::get<1>(gate).first}, {"key", std::get<1>(gate).second}});
        }
    }

    friend void from_json(const Json& j, Circuit& circuit) {
        circuit = Circuit(j.at("n_qubits").get<std::uint64_t>());
        const Json& tmp_list = j.at("gate_list");
        for (const Json& gate_with_key : tmp_list) {
            if (gate_with_key.contains("key")) {
                circuit.add_param_gate(gate_with_key.at("gate").get<ParamGate<Fp, Sp>>(),
                                       gate_with_key.at("key").get<std::string>());
            } else {
                circuit.add_gate(gate_with_key.at("gate").get<Gate<Fp, Sp>>());
            }
        }
    }

private:
    std::uint64_t _n_qubits;

    std::vector<GateWithKey> _gate_list;

    void check_gate_is_valid(const Gate<Fp, Sp>& gate) const;

    void check_gate_is_valid(const ParamGate<Fp, Sp>& gate) const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp, ExecutionSpace Sp>
void bind_circuit_circuit_hpp(nb::module_& m) {
    nb::class_<Circuit<Fp, Sp>>(m, "Circuit", "Quantum circuit represented as gate array")
        .def(nb::init<std::uint64_t>(), "Initialize empty circuit of specified qubits.")
        .def("n_qubits", &Circuit<Fp, Sp>::n_qubits, "Get property of `n_qubits`.")
        .def("gate_list",
             &Circuit<Fp, Sp>::gate_list,
             "Get property of `gate_list`.",
             nb::rv_policy::reference)
        .def("n_gates", &Circuit<Fp, Sp>::n_gates, "Get property of `n_gates`.")
        .def("key_set", &Circuit<Fp, Sp>::key_set, "Get set of keys of parameters.")
        .def("get_gate_at", &Circuit<Fp, Sp>::get_gate_at, "Get reference of i-th gate.")
        .def("get_param_key_at",
             &Circuit<Fp, Sp>::get_param_key_at,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit<Fp, Sp>::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate<Fp, Sp>&>(&Circuit<Fp, Sp>::add_gate),
             "Add gate. Given gate is copied.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate<Fp, Sp>&, std::string_view>(
                 &Circuit<Fp, Sp>::add_param_gate),
             "Add parametric gate with specifing key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit<Fp, Sp>&>(&Circuit<Fp, Sp>::add_circuit),
             "Add all gates in specified circuit. Given gates are copied.")
        .def("update_quantum_state",
             &Circuit<Fp, Sp>::update_quantum_state,
             "Apply gate to the StateVector. StateVector in args is directly updated. If the "
             "circuit contains parametric gate, you have to give real value of parameter as "
             "dict[str, float] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Fp, Sp>& circuit, StateVector<Fp, Sp>& state, nb::kwargs kwargs) {
                std::map<std::string, Fp> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<Fp>(param);
                }
                circuit.update_quantum_state(state, parameters);
            },
            "Apply gate to the StateVector. StateVector in args is directly updated. If the "
            "circuit contains parametric gate, you have to give real value of parameter as "
            "\"name=value\" format in kwargs.")
        .def("update_quantum_state",
             [](const Circuit<Fp>& circuit, StateVector<Fp, Sp>& state) {
                 circuit.update_quantum_state(state);
             })
        .def("copy", &Circuit<Fp, Sp>::copy, "Copy circuit. All the gates inside is copied.")
        .def("get_inverse",
             &Circuit<Fp, Sp>::get_inverse,
             "Get inverse of circuit. All the gates are newly created.")
        .def(
            "to_json",
            [](const Circuit<Fp, Sp>& circuit) { return Json(circuit).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](Circuit<Fp, Sp>& circuit, const std::string& str) {
                circuit = nlohmann::json::parse(str);
            },
            "Read an object from the JSON representation of the circuit.");
}
}  // namespace internal
#endif
}  // namespace scaluq
