#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace qulacs {
using Gate = std::unique_ptr<class QuantumGate>;

class QuantumGate {
public:
    [[nodiscard]] virtual std::vector<UINT> get_target_qubit_list() const = 0;
    [[nodiscard]] virtual std::vector<UINT> get_control_qubit_list() const = 0;
    [[nodiscard]] UINT n_qubits() const {
        auto target_list = get_target_qubit_list();
        auto control_list = get_control_qubit_list();
        UINT ret = 0;
        if (!target_list.empty()) ret = std::max(ret, *std::ranges::max_element(target_list) + 1);
        if (!control_list.empty()) ret = std::max(ret, *std::ranges::max_element(control_list) + 1);
        return ret;
    }

    [[nodiscard]] virtual Gate copy() const;
    [[nodiscard]] virtual Gate get_inverse() const;

    virtual void update_quantum_state(StateVector& state_vector) const = 0;
};

}  // namespace qulacs
