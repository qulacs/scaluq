#pragma once

#include <set>
#include <variant>

#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../types.hpp"

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
class Circuit {
public:
    using GateWithKey =
        std::variant<Gate<Prec, Space>, std::pair<ParamGate<Prec, Space>, std::string>>;
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

    void add_gate(const Gate<Prec, Space>& gate) {
        check_gate_is_valid(gate);
        _gate_list.push_back(gate);
    }
    void add_param_gate(const ParamGate<Prec, Space>& param_gate, std::string_view parameter_key) {
        check_gate_is_valid(param_gate);
        _gate_list.push_back(std::make_pair(param_gate, std::string(parameter_key)));
    }

    void add_circuit(const Circuit<Prec, Space>& circuit);
    void add_circuit(Circuit<Prec, Space>&& circuit);

    void update_quantum_state(StateVector<Prec, Space>& state,
                              const std::map<std::string, double>& parameters = {}) const;
    void update_quantum_state(
        StateVectorBatched<Prec, Space>& states,
        const std::map<std::string, std::vector<double>>& parameters = {}) const;

    Circuit copy() const;

    Circuit get_inverse() const;

    void optimize(std::uint64_t max_block_size = 3);

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
                circuit.add_param_gate(gate_with_key.at("gate").get<ParamGate<Prec, Space>>(),
                                       gate_with_key.at("key").get<std::string>());
            } else {
                circuit.add_gate(gate_with_key.at("gate").get<Gate<Prec, Space>>());
            }
        }
    }

    /**
     * @brief サンプリングされうるすべてのパターンに対して，それぞれが何回選ばれたかを返す
     * @attention ProbabilisticGate に ProbabilisticGate が含まれてはいけない
     */
    std::vector<std::pair<StateVector<Prec, Space>, std::int64_t>> simulate_noise(
        const StateVector<Prec, Space>& initial_state,
        std::uint64_t sampling_count,
        const std::map<std::string, double>& parameters = {},
        std::uint64_t seed = 0) const;

private:
    std::uint64_t _n_qubits;

    std::vector<GateWithKey> _gate_list;

    void check_gate_is_valid(const Gate<Prec, Space>& gate) const;

    void check_gate_is_valid(const ParamGate<Prec, Space>& gate) const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_circuit_circuit_hpp(nb::module_& m) {
    nb::class_<Circuit<Prec, Space>>(m,
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
        .def("n_qubits", &Circuit<Prec, Space>::n_qubits, "Get property of `n_qubits`.")
        .def("gate_list",
             &Circuit<Prec, Space>::gate_list,
             "Get property of `gate_list`.",
             nb::rv_policy::reference)
        .def("n_gates", &Circuit<Prec, Space>::n_gates, "Get property of `n_gates`.")
        .def("key_set", &Circuit<Prec, Space>::key_set, "Get set of keys of parameters.")
        .def("get_gate_at",
             &Circuit<Prec, Space>::get_gate_at,
             "index"_a,
             "Get reference of i-th gate.")
        .def("get_param_key_at",
             &Circuit<Prec, Space>::get_param_key_at,
             "index"_a,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit<Prec, Space>::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate<Prec, Space>&>(&Circuit<Prec, Space>::add_gate),
             "gate"_a,
             "Add gate. Given gate is copied.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate<Prec, Space>&, std::string_view>(
                 &Circuit<Prec, Space>::add_param_gate),
             "param_gate"_a,
             "param_key"_a,
             "Add parametric gate with specifying key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit<Prec, Space>&>(&Circuit<Prec, Space>::add_circuit),
             "other"_a,
             "Add all gates in specified circuit. Given gates are copied.")
        .def("update_quantum_state",
             nb::overload_cast<StateVector<Prec, Space>&, const std::map<std::string, double>&>(
                 &Circuit<Prec, Space>::update_quantum_state, nb::const_),
             "state"_a,
             "params"_a,
             "Apply gate to the StateVector. StateVector in args is directly updated. If the "
             "circuit contains parametric gate, you have to give real value of parameter as "
             "dict[str, float] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Prec, Space>& circuit,
                StateVector<Prec, Space>& state,
                nb::kwargs kwargs) {
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
            nb::overload_cast<StateVectorBatched<Prec, Space>&,
                              const std::map<std::string, std::vector<double>>&>(
                &Circuit<Prec, Space>::update_quantum_state, nb::const_),
            "state"_a,
            "params"_a,
            "Apply gate to the StateVectorBatched. StateVectorBatched in args is directly updated. "
            "If the circuit contains parametric gate, you have to give real value of parameter as "
            "dict[str, list[float]] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit<Prec, Space>& circuit,
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
        .def("copy",
             &Circuit<Prec, Space>::copy,
             "Copy circuit. Returns a new circuit instance with all gates copied by reference.")
        .def("get_inverse",
             &Circuit<Prec, Space>::get_inverse,
             "Get inverse of circuit. All the gates are newly created.")
        .def("optimize",
             &Circuit<Prec, Space>::optimize,
             "max_block_size"_a = 3,
             "Optimize circuit. Create qubit dependency tree and merge neighboring gates if the "
             "new gate has less than or equal to `max_block_size` or the new gate is Pauli.")
        .def("simulate_noise",
             &Circuit<Prec, Space>::simulate_noise,
             "Simulate noise circuit. Return all the possible states and their counts.")
        .def(
            "to_json",
            [](const Circuit<Prec, Space>& circuit) { return Json(circuit).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](Circuit<Prec, Space>& circuit, const std::string& str) {
                circuit = nlohmann::json::parse(str);
            },
            "json_str"_a,
            "Read an object from the JSON representation of the circuit.");
}
}  // namespace internal
#endif
}  // namespace scaluq
