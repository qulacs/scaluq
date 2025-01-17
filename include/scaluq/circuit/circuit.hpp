#pragma once

#include <set>
#include <variant>

#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../types.hpp"

namespace scaluq {

template <Precision Prec>
class Circuit {
public:
    using GateWithKey = std::variant<Gate<Prec>, std::pair<ParamGate<Prec>, std::string>>;
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

    void add_gate(const Gate<Prec>& gate) {
        check_gate_is_valid(gate);
        _gate_list.push_back(gate);
    }
    void add_param_gate(const ParamGate<Prec>& param_gate, std::string_view parameter_key) {
        check_gate_is_valid(param_gate);
        _gate_list.push_back(std::make_pair(param_gate, std::string(parameter_key)));
    }

    void add_circuit(const Circuit<Prec>& circuit);
    void add_circuit(Circuit<Prec>&& circuit);

    void update_quantum_state(StateVector<Prec>& state,
                              const std::map<std::string, double>& parameters = {}) const;

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
                circuit.add_param_gate(gate_with_key.at("gate").get<ParamGate<Prec>>(),
                                       gate_with_key.at("key").get<std::string>());
            } else {
                circuit.add_gate(gate_with_key.at("gate").get<Gate<Prec>>());
            }
        }
    }

private:
    std::uint64_t _n_qubits;

    std::vector<GateWithKey> _gate_list;

    void check_gate_is_valid(const Gate<Prec>& gate) const;

    void check_gate_is_valid(const ParamGate<Prec>& gate) const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_circuit_circuit_hpp(nb::module_& m) {
    nb::class_<Circuit<Prec>>(m, "Circuit", "Quantum circuit represented as gate array")
        .def(nb::init<std::uint64_t>(), "Initialize empty circuit of specified qubits.")
        .def("n_qubits", &Circuit<Prec>::n_qubits, "Get property of `n_qubits`.")
        .def("gate_list",
             &Circuit<Prec>::gate_list,
             "Get property of `gate_list`.",
             nb::rv_policy::reference)
        .def("n_gates", &Circuit<Prec>::n_gates, "Get property of `n_gates`.")
        .def("key_set", &Circuit<Prec>::key_set, "Get set of keys of parameters.")
        .def("get_gate_at", &Circuit<Prec>::get_gate_at, "Get reference of i-th gate.")
        .def("get_param_key_at",
             &Circuit<Prec>::get_param_key_at,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit<Prec>::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate<Prec>&>(&Circuit<Prec>::add_gate),
             "Add gate. Given gate is copied.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate<Prec>&, std::string_view>(
                 &Circuit<Prec>::add_param_gate),
             "Add parametric gate with specifing key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit<Prec>&>(&Circuit<Prec>::add_circuit),
             "Add all gates in specified circuit. Given gates are copied.")
        .def("update_quantum_state",
             &Circuit<Prec>::update_quantum_state,
             "Apply gate to the StateVector. StateVector in args is directly updated. If the "
             "circuit contains parametric gate, you have to give real value of parameter as "
             "dict[str, float] in 2nd arg.", )
        .def(
            "update_quantum_state",
            &Circuit < Prec nb::overload_cast <
                [&](const Circuit<Prec>& circuit, StateVector<Prec>& state, nb::kwargs kwargs) {
                    std::map<std::string, double> parameters;
                    for (auto&& [key, param] : kwargs) {
                        parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
                    }
                    circuit.update_quantum_state(state, parameters);
                },
            "Apply gate to the StateVector. StateVector in args is directly updated. If the "
            "circuit contains parametric gate, you have to give real value of parameter as "
            "\"name=value\" format in kwargs.")
        .def("copy", &Circuit<Prec>::copy, "Copy circuit. All the gates inside is copied.")
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
            "Read an object from the JSON representation of the circuit.");
}
}  // namespace internal
#endif
}  // namespace scaluq
