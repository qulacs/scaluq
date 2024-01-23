#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace qulacs {
namespace internal {
// forward declarations
class GateBase;

template <typename T>
concept GateImpl = std::derived_from<T, GateBase>;

template <GateImpl T>
class GatePtr;
}  // namespace internal
using Gate = internal::GatePtr<internal::GateBase>;

namespace internal {
class GateBase {
public:
    virtual ~GateBase() = default;

    [[nodiscard]] virtual std::vector<UINT> get_target_qubit_list() const = 0;
    [[nodiscard]] virtual std::vector<UINT> get_control_qubit_list() const = 0;

    [[nodiscard]] virtual Gate copy() const = 0;
    [[nodiscard]] virtual Gate get_inverse() const = 0;

    virtual void update_quantum_state(StateVector& state_vector) const = 0;
};

template <GateImpl T>
class GatePtr {
    friend class GateFactory;

private:
    std::shared_ptr<internal::GateBase> _gate_ptr;

public:
    template <GateImpl U>
    GatePtr(const std::shared_ptr<U>& gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _gate_ptr = gate_ptr;
        } else if constexpr (std::is_same_v<U, internal::GateBase>) {
            _gate_ptr = std::static_pointer_cast<T>(gate_ptr);
        } else {
            if (!(_gate_ptr = std::dynamic_pointer_cast<T>(gate_ptr))) {
                throw std::runtime_error("invalid gate cast");
            }
        }
    }
    template <GateImpl U>
    GatePtr(const GatePtr& gate) : GatePtr(gate->_gate_ptr) {}

    GateBase* operator->() const { return _gate_ptr.get(); }
};
}  // namespace internal

}  // namespace qulacs
