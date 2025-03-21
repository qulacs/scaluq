#pragma once

#include <set>
#include <variant>

#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../types.hpp"

namespace scaluq {

template <std::floating_point Fp>
class Circuit {
public:
    using GateWithKey = std::variant<Gate<Fp>, std::pair<ParamGate<Fp>, std::string>>;
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

    void add_gate(const Gate<Fp>& gate) {
        check_gate_is_valid(gate);
        _gate_list.push_back(gate);
    }
    void add_gate(Gate<Fp>&& gate) {
        check_gate_is_valid(gate);
        _gate_list.push_back(std::move(gate));
    }
    void add_param_gate(const ParamGate<Fp>& param_gate, std::string_view parameter_key) {
        check_gate_is_valid(param_gate);
        _gate_list.push_back(std::make_pair(param_gate, std::string(parameter_key)));
    }
    void add_param_gate(ParamGate<Fp>&& param_gate, std::string_view parameter_key) {
        check_gate_is_valid(param_gate);
        _gate_list.push_back(std::make_pair(std::move(param_gate), std::string(parameter_key)));
    }

    void add_circuit(const Circuit<Fp>& circuit);
    void add_circuit(Circuit<Fp>&& circuit);

    void update_quantum_state(StateVector<Fp>& state,
                              const std::map<std::string, Fp>& parameters = {}) const;
    void update_quantum_state(StateVectorBatched<Fp>& states,
                              const std::map<std::string, std::vector<Fp>>& parameters = {}) const;

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
                circuit.add_param_gate(gate_with_key.at("gate").get<ParamGate<Fp>>(),
                                       gate_with_key.at("key").get<std::string>());
            } else {
                circuit.add_gate(gate_with_key.at("gate").get<Gate<Fp>>());
            }
        }
    }

private:
    std::uint64_t _n_qubits;

    std::vector<GateWithKey> _gate_list;

    void check_gate_is_valid(const Gate<Fp>& gate) const;

    void check_gate_is_valid(const ParamGate<Fp>& gate) const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_circuit_circuit_hpp(nb::module_& m) {
    nb::class_<Circuit<Fp>>(m,
                            "Circuit",
                            DocString()
                                .desc("Quantum circuit representation.")
                                .arg("n_qubits", "Number of qubits in the circuit.")
                                .ex(DocString::Code({">>> circuit = Circuit(3)",
                                                     ">>> print(circuit.to_json())",
                                                     "{\"gate_list\":[],\"n_qubits\":3}"}))
                                .build_as_google_style()
                                .c_str())
        .def(nb::init<std::uint64_t>(),
             "n_qubits"_a,
             "Initialize empty circuit of specified qubits.")
        .def("n_qubits", &Circuit<Fp>::n_qubits, "Get property of `n_qubits`.")
        .def("gate_list",
             &Circuit<Fp>::gate_list,
             "Get property of `gate_list`.",
             nb::rv_policy::reference)
        .def("n_gates", &Circuit<Fp>::n_gates, "Get property of `n_gates`.")
        .def("key_set", &Circuit<Fp>::key_set, "Get set of keys of parameters.")
        .def("get_gate_at", &Circuit<Fp>::get_gate_at, "index"_a, "Get reference of i-th gate.")
        .def("get_param_key_at",
             &Circuit<Fp>::get_param_key_at,
             "index"_a,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit<Fp>::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate<Fp>&>(&Circuit<Fp>::add_gate),
             "gate"_a,
             "Add gate. Given gate is copied.")
        .def(
            "add_param_gate",
            nb::overload_cast<const ParamGate<Fp>&, std::string_view>(&Circuit<Fp>::add_param_gate),
            "param_gate"_a,
            "param_key"_a,
            "Add parametric gate with specifing key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit<Fp>&>(&Circuit<Fp>::add_circuit),
             "other"_a,
             "Add all gates in specified circuit. Given gates are copied.")
        .def("update_quantum_state",
             nb::overload_cast<StateVector<Fp>&, const std::map<std::string, Fp>&>(
                 &Circuit<Fp>::update_quantum_state, nb::const_),
             "state"_a,
             "params"_a,
             "Apply gate to the StateVector. StateVector in args is directly updated. If the "
             "circuit contains parametric gate, you have to give real value of parameter as "
             "dict[str, float] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Fp>& circuit, StateVector<Fp>& state, nb::kwargs kwargs) {
                std::map<std::string, Fp> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<Fp>(param);
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
            nb::overload_cast<StateVectorBatched<Fp>&,
                              const std::map<std::string, std::vector<Fp>>&>(
                &Circuit<Fp>::update_quantum_state, nb::const_),
            "state"_a,
            "params"_a,
            "Apply gate to the StateVectorBatched. StateVectorBatched in args is directly updated. "
            "If the circuit contains parametric gate, you have to give real value of parameter as "
            "dict[str, list[float]] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Fp>& circuit, StateVectorBatched<Fp>& states, nb::kwargs kwargs) {
                std::map<std::string, std::vector<Fp>> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<std::vector<Fp>>(param);
                }
                circuit.update_quantum_state(states, parameters);
            },
            "state"_a,
            "kwargs"_a,
            "Apply gate to the StateVectorBatched. StateVectorBatched in args is directly updated. "
            "If the circuit contains parametric gate, you have to give real value of parameter as "
            "\"name=[value1, value2, ...]\" format in kwargs.")
        .def("copy",
             &Circuit<Fp>::copy,
             "Copy circuit. Returns a new circuit instance with all gates copied by reference.")
        .def("get_inverse",
             &Circuit<Fp>::get_inverse,
             "Get inverse of circuit. All the gates are newly created.")
        .def(
            "to_json",
            [](const Circuit<Fp>& circuit) { return Json(circuit).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](Circuit<Fp>& circuit, const std::string& str) {
                circuit = nlohmann::json::parse(str);
            },
            "json_str"_a,
            "Read an object from the JSON representation of the circuit.");
}
}  // namespace internal
#endif
}  // namespace scaluq
