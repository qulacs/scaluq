#pragma once

#include <set>
#include <variant>

#include "../gate/gate.hpp"
#include "../gate/pgate.hpp"
#include "../types.hpp"

namespace scaluq {
class Circuit {
public:
    using GateWithKey = std::variant<Gate, std::pair<PGate, std::string_view>>;
    explicit Circuit(UINT n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline UINT n_qubits() const { return _n_qubits; }
    [[nodiscard]] inline const std::vector<GateWithKey>& gate_list() const { return _gate_list; }
    [[nodiscard]] inline UINT gate_count() { return _gate_list.size(); }
    [[nodiscard]] inline const std::set<std::string_view> key_set() const {
        std::set<std::string_view> key_set;
        for (auto&& gate : _gate_list) {
            if (gate.index() == 1) key_set.insert(std::get<1>(gate).second);
        }
        return key_set;
    }
    [[nodiscard]] inline const GateWithKey& get(UINT idx) const {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error("Circuit::get(UINT): index out of bounds");
        }
        return _gate_list[idx];
    }
    [[nodiscard]] inline std::optional<std::string_view> get_key(UINT idx) {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error("Circuit::get_parameter_key(UINT): index out of bounds");
        }
        const auto& gate = _gate_list[idx];
        if (gate.index() == 0) return std::nullopt;
        return std::get<1>(gate).second;
    }

    [[nodiscard]] UINT calculate_depth() const;

    void add_gate(const Gate& gate);
    void add_gate(Gate&& gate);
    void add_pgate(const PGate& pgate, std::string_view key);
    void add_pgate(PGate&& pgate, std::string_view key);
    void add_circuit(const Circuit& circuit);
    void add_circuit(Circuit&& circuit);

    void update_quantum_state(StateVector& state,
                              const std::map<std::string_view, double>& parameters = {}) const;

    Circuit copy() const;
    Circuit get_inverse() const;

private:
    UINT _n_qubits;

    std::vector<GateWithKey> _gate_list;

    void check_gate_is_valid(const Gate& gate) const;
    void check_gate_is_valid(const PGate& gate) const;
};
}  // namespace scaluq
