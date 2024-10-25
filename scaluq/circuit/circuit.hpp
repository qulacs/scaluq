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

    [[nodiscard]] std::uint64_t calculate_depth() const {
        std::vector<std::uint64_t> filled_step(_n_qubits, 0ULL);
        for (const auto& gate : _gate_list) {
            std::vector<std::uint64_t> control_qubits =
                gate.index() == 0 ? std::get<0>(gate)->control_qubit_list()
                                  : std::get<1>(gate).first->control_qubit_list();
            std::vector<std::uint64_t> target_qubits =
                gate.index() == 0 ? std::get<0>(gate)->target_qubit_list()
                                  : std::get<1>(gate).first->target_qubit_list();
            std::uint64_t max_step_amount_target_qubits = 0;
            for (std::uint64_t control : control_qubits) {
                if (max_step_amount_target_qubits < filled_step[control]) {
                    max_step_amount_target_qubits = filled_step[control];
                }
            }
            for (std::uint64_t target : control_qubits) {
                if (max_step_amount_target_qubits < filled_step[target]) {
                    max_step_amount_target_qubits = filled_step[target];
                }
            }
            for (std::uint64_t control : control_qubits) {
                filled_step[control] = max_step_amount_target_qubits + 1;
            }
            for (std::uint64_t target : target_qubits) {
                filled_step[target] = max_step_amount_target_qubits + 1;
            }
        }
        return *std::ranges::max_element(filled_step);
    }

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

    void add_circuit(const Circuit<Fp>& circuit) {
        if (circuit._n_qubits != _n_qubits) {
            throw std::runtime_error(
                "Circuit::add_circuit(const Circuit&): circuit with different qubit count cannot "
                "be "
                "merged.");
        }
        _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
        for (const auto& gate : circuit._gate_list) {
            _gate_list.push_back(gate);
        }
    }
    void add_circuit(Circuit<Fp>&& circuit) {
        if (circuit._n_qubits != _n_qubits) {
            throw std::runtime_error(
                "Circuit::add_circuit(Circuit&&): circuit with different qubit count cannot be "
                "merged.");
        }
        _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
        for (auto&& gate : circuit._gate_list) {
            _gate_list.push_back(std::move(gate));
        }
    }

    void update_quantum_state(StateVector<Fp>& state,
                              const std::map<std::string, Fp>& parameters = {}) const {
        for (auto&& gate : _gate_list) {
            if (gate.index() == 0) continue;
            const auto& key = std::get<1>(gate).second;
            if (!parameters.contains(key)) {
                using namespace std::string_literals;
                throw std::runtime_error(
                    "Circuit::update_quantum_state(StateVector&, const std::map<std::string_view, double>&) const: parameter named "s +
                    std::string(key) + "is not given.");
            }
        }
        for (auto&& gate : _gate_list) {
            if (gate.index() == 0) {
                std::get<0>(gate)->update_quantum_state(state);
            } else {
                const auto& [param_gate, key] = std::get<1>(gate);
                param_gate->update_quantum_state(state, parameters.at(key));
            }
        }
    }

    Circuit copy() const {
        Circuit ccircuit(_n_qubits);
        ccircuit._gate_list.reserve(_gate_list.size());
        for (auto&& gate : _gate_list) {
            if (gate.index() == 0)
                ccircuit._gate_list.push_back(std::get<0>(gate));
            else {
                const auto& [param_gate, key] = std::get<1>(gate);
                ccircuit._gate_list.push_back(std::make_pair(param_gate, key));
            }
        }
        return ccircuit;
    }

    Circuit get_inverse() const {
        Circuit icircuit(_n_qubits);
        icircuit._gate_list.reserve(_gate_list.size());
        for (auto&& gate : _gate_list | std::views::reverse) {
            if (gate.index() == 0)
                icircuit._gate_list.push_back(std::get<0>(gate)->get_inverse());
            else {
                const auto& [param_gate, key] = std::get<1>(gate);
                icircuit._gate_list.push_back(std::make_pair(param_gate->get_inverse(), key));
            }
        }
        return icircuit;
    }

private:
    std::uint64_t _n_qubits;

    std::vector<GateWithKey> _gate_list;

    void check_gate_is_valid(const Gate<Fp>& gate) const {
        auto targets = gate->target_qubit_list();
        auto controls = gate->control_qubit_list();
        bool valid = true;
        if (!targets.empty())
            valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
        if (!controls.empty())
            valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
        if (!valid) {
            throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
        }
    }

    void check_gate_is_valid(const ParamGate<Fp>& gate) const {
        auto targets = gate->target_qubit_list();
        auto controls = gate->control_qubit_list();
        bool valid = true;
        if (!targets.empty())
            valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
        if (!controls.empty())
            valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
        if (!valid) {
            throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
        }
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_circuit_circuit_hpp(nb::module_& m) {
    nb::class_<Circuit<double>>(m, "Circuit", "Quantum circuit represented as gate array")
        .def(nb::init<std::uint64_t>(), "Initialize empty circuit of specified qubits.")
        .def("n_qubits", &Circuit<double>::n_qubits, "Get property of `n_qubits`.")
        .def("gate_list",
             &Circuit<double>::gate_list,
             "Get property of `gate_list`.",
             nb::rv_policy::reference)
        .def("n_gates", &Circuit<double>::n_gates, "Get property of `n_gates`.")
        .def("key_set", &Circuit<double>::key_set, "Get set of keys of parameters.")
        .def("get_gate_at", &Circuit<double>::get_gate_at, "Get reference of i-th gate.")
        .def("get_param_key_at",
             &Circuit<double>::get_param_key_at,
             "Get parameter key of i-th gate. If it is not parametric, return None.")
        .def("calculate_depth", &Circuit<double>::calculate_depth, "Get depth of circuit.")
        .def("add_gate",
             nb::overload_cast<const Gate<double>&>(&Circuit<double>::add_gate),
             "Add gate. Given gate is copied.")
        .def("add_param_gate",
             nb::overload_cast<const ParamGate<double>&, std::string_view>(
                 &Circuit<double>::add_param_gate),
             "Add parametric gate with specifing key. Given param_gate is copied.")
        .def("add_circuit",
             nb::overload_cast<const Circuit<double>&>(&Circuit<double>::add_circuit),
             "Add all gates in specified circuit. Given gates are copied.")
        .def("update_quantum_state",
             &Circuit<double>::update_quantum_state,
             "Apply gate to the StateVector. StateVector in args is directly updated. If the "
             "circuit contains parametric gate, you have to give real value of parameter as "
             "dict[str, float] in 2nd arg.")
        .def(
            "update_quantum_state",
            [&](const Circuit<double>& circuit, StateVector<double>& state, nb::kwargs kwargs) {
                std::map<std::string, double> parameters;
                for (auto&& [key, param] : kwargs) {
                    parameters[nb::cast<std::string>(key)] = nb::cast<double>(param);
                }
                circuit.update_quantum_state(state, parameters);
            },
            "Apply gate to the StateVector. StateVector in args is directly updated. If the "
            "circuit contains parametric gate, you have to give real value of parameter as "
            "\"name=value\" format in kwargs.")
        .def("update_quantum_state",
             [](const Circuit<double>& circuit, StateVector<double>& state) {
                 circuit.update_quantum_state(state);
             })
        .def("copy", &Circuit<double>::copy, "Copy circuit. All the gates inside is copied.")
        .def("get_inverse",
             &Circuit<double>::get_inverse,
             "Get inverse of circuit. All the gates are newly created.");
}
}  // namespace internal
#endif
}  // namespace scaluq
