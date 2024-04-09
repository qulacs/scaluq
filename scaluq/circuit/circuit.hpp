#pragma once

#include "../gate/gate.hpp"
#include "../types.hpp"

namespace scaluq {
class Circuit {
public:
    explicit Circuit(UINT n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline UINT n_qubits() const { return _n_qubits; }
    [[nodiscard]] inline const std::vector<Gate>& gate_list() const { return _gate_list; }
    [[nodiscard]] inline UINT gate_count() { return _gate_list.size(); }
    [[nodiscard]] inline const Gate& get(UINT idx) const {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error("Circuit::get(UINT): index out of bounds");
        }
        return _gate_list[idx];
    }
    [[nodiscard]] inline Gate& get(UINT idx) {
        if (idx >= _gate_list.size()) {
            throw std::runtime_error("Circuit::get(UINT): index out of bounds");
        }
        return _gate_list[idx];
    }

    [[nodiscard]] UINT calculate_depth() const;

    void add_gate(const Gate& gate);
    void add_gate(Gate&& gate);
    void add_circuit(const Circuit& circuit);
    void add_circuit(Circuit&& circuit);

    void update_quantum_state(StateVector& state) const;

    Circuit copy() const;
    Circuit get_inverse() const;

private:
    UINT _n_qubits;

    std::vector<Gate> _gate_list;

    void check_gate_is_valid(const Gate& gate) const;
};
}  // namespace scaluq
