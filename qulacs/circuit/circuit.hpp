#pragma once

#include "../gate/gate.hpp"
#include "../types.hpp"

namespace qulacs {
class Circuit {
public:
    explicit Circuit(UINT n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] UINT n_qubits() const { return _n_qubits; }

    void add_gate(Gate gate);

    void update_quantum_state(StateVector& state) const;

    Circuit copy() const;
    Circuit get_inverse() const;

private:
    UINT _n_qubits;

    std::vector<std::unique_ptr<QuantumGate>> _gate_list;

    void check_gate_is_valid(const QuantumGate& gate) const;
};
}  // namespace qulacs
