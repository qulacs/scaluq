#pragma once

#include <vector>

#include "gate.hpp"

namespace qulacs {
class Pauli : public QuantumGate {
    std::vector<UINT> target_qubit_index_list;
    std::vector<UINT> pauli_id_list;

public:
    Pauli(std::vector<UINT> target_list, std::vector<UINT> pauli_list)
        : target_qubit_index_list(target_list), pauli_id_list(pauli_list) {}
    void update_quantum_state(StateVector& state_vector) const override;
};

class PauliRotation : public QuantumGate {
    std::vector<UINT> target_qubit_index_list;
    std::vector<UINT> pauli_id_list;
    double angle;

public:
    PauliRotation(std::vector<UINT> target_list, std::vector<UINT> pauli_list, double angle)
        : target_qubit_index_list(target_list), pauli_id_list(pauli_list), angle(angle) {}
    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace qulacs
