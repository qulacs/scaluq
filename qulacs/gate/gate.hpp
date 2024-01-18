#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace qulacs {
namespace internal {
using GatePtr = std::unique_ptr<class GateBase>;
}

namespace internal {
class GateBase {
public:
    virtual ~GateBase() = default;

    [[nodiscard]] virtual std::vector<UINT> get_target_qubit_list() const = 0;
    [[nodiscard]] virtual std::vector<UINT> get_control_qubit_list() const = 0;

    [[nodiscard]] virtual GatePtr copy() const = 0;
    [[nodiscard]] virtual GatePtr get_inverse() const = 0;

    virtual void update_quantum_state(StateVector& state_vector) const = 0;
};

template <typename T>
concept GateImpl = std::derived_from<T, GateBase>;

class GateFactory;
}  // namespace internal

class Gate {
    friend class internal::GateFactory;

private:
    internal::GatePtr _gate_ptr;
    Gate(internal::GatePtr&& gate_ptr) : _gate_ptr(std::move(gate_ptr)) {}

public:
    std::vector<UINT> get_control_qubit_list() const { return _gate_ptr->get_control_qubit_list(); }
    std::vector<UINT> get_target_qubit_list() const { return _gate_ptr->get_target_qubit_list(); }
    Gate copy() const { return _gate_ptr->copy(); }
    Gate get_inverse() const { return _gate_ptr->get_inverse(); }
    void update_quantum_state(StateVector& state_vector) const {
        _gate_ptr->update_quantum_state(state_vector);
    }
};

}  // namespace qulacs
