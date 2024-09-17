#pragma once

#include <set>
#include <variant>

#include "../gate/gate.hpp"
#include "../gate/param_gate.hpp"
#include "../types.hpp"

namespace scaluq {
class Circuit {
public:
    using GateWithKey = std::variant<Gate, std::pair<ParamGate, std::string>>;
    explicit Circuit(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline std::uint64_t n_qubits() const { return _n_qubits; }
    [[nodiscard]] inline const std::vector<GateWithKey>& gate_list() const { return _gate_list; }
    [[nodiscard]] inline std::uint64_t n_gates() { return _gate_list.size(); }
    [[nodiscard]] inline const std::set<std::string> key_set() const {
        std::set<std::string> key_set;
        for (auto&& gate : _gate_list) {
            if (gate.index() == 1) key_set.insert(std::get<1>(gate).second);
        }
        return key_set;
    }
    [[nodiscard]] inline const GateWithKey& get_gate_at(std::uint64_t idx) const {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error("Circuit::get_gate_at(std::uint64_t): index out of bounds");
        }
        return _gate_list[idx];
    }
    [[nodiscard]] inline std::optional<std::string> get_param_key_at(std::uint64_t idx) {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error(
                "Circuit::get_parameter_key(std::uint64_t): index out of bounds");
        }
        const auto& gate = _gate_list[idx];
        if (gate.index() == 0) return std::nullopt;
        return std::get<1>(gate).second;
    }

    [[nodiscard]] std::uint64_t calculate_depth() const;

    void add_gate(const Gate& gate);
    void add_gate(Gate&& gate);
    void add_param_gate(const ParamGate& param_gate, std::string_view key);
    void add_param_gate(ParamGate&& param_gate, std::string_view key);
    void add_circuit(const Circuit& circuit);
    void add_circuit(Circuit&& circuit);

    void update_quantum_state(StateVector& state,
                              const std::map<std::string, double>& parameters = {}) const;

    Circuit copy() const;
    Circuit get_inverse() const;

private:
    std::uint64_t _n_qubits;

    std::vector<GateWithKey> _gate_list;

    void check_gate_is_valid(const Gate& gate) const;
    void check_gate_is_valid(const ParamGate& gate) const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_circuit_circuit_hpp(nb::module_& m) {
    nb::class_<Circuit>(m, "Circuit", "Quantum circuit represented as gate array")
        .def(nb::init<std::uint64_t>(), "Initialize empty circuit of specified qubits.")
        .def("n_qubits", &Circuit::n_qubits, "Get property of `n_qubits`.")
        .def("gate_list",
             &Circuit::gate_list,
             "Get property of `gate_list`.",
             nb::rv_policy::reference)
        .def("n_gates", &Circuit::n_gates, "Get property of `n_gates`.")
        .def("key_set", &Circuit::key_set, "Get set of keys of parameters.")
        .def("get_gate_at", &Circuit::get_gate_at, "Get reference of i-th gate.")
        .def("get_param_key_at",
             &Circuit::get_param_key_at,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate&>(&Circuit::add_gate),
             "Add gate. Given gate is copied.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate&, std::string_view>(&Circuit::add_param_gate),
             "Add parametric gate with specifing key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit&>(&Circuit::add_circuit),
             "Add all gates in specified circuit. Given gates are copied.")
        .def("update_quantum_state",
             &Circuit::update_quantum_state,
             "Apply gate to the StateVector. StateVector in args is directly updated. If the "
             "circuit contains parametric gate, you have to give real value of parameter as "
             "dict[str, float] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit& circuit, StateVector& state, nb::kwargs kwargs) {
                std::map<std::string, double> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
                }
                circuit.update_quantum_state(state, parameters);
            },
            "Apply gate to the StateVector. StateVector in args is directly updated. If the "
            "circuit contains parametric gate, you have to give real value of parameter as "
            "\"name=value\" format in kwargs.")
        .def(
            "update_quantum_state",
            [](const Circuit& circuit, StateVector& state) { circuit.update_quantum_state(state); })
        .def("copy", &Circuit::copy, "Copy circuit. All the gates inside is copied.")
        .def("get_inverse",
             &Circuit::get_inverse,
             "Get inverse of circuit. All the gates are newly created.");
}
}  // namespace internal
#endif
}  // namespace scaluq
