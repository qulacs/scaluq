#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
// forward declarations
class GateBase;

template <typename T>
concept GateImpl = std::derived_from<T, GateBase>;

class IGateImpl;
class GlobalPhaseGateImpl;
class XGateImpl;
class YGateImpl;
class ZGateImpl;
class HGateImpl;
class SGateImpl;
class SdagGateImpl;
class TGateImpl;
class TdagGateImpl;
class SqrtXGateImpl;
class SqrtXdagGateImpl;
class SqrtYGateImpl;
class SqrtYdagGateImpl;
class P0GateImpl;
class P1GateImpl;
class RXGateImpl;
class RYGateImpl;
class RZGateImpl;
class U1GateImpl;
class U2GateImpl;
class U3GateImpl;
class OneTargetMatrixGateImpl;
class CXGateImpl;
class CZGateImpl;
class CCXGateImpl;
class SwapGateImpl;
class TwoTargetMatrixGateImpl;
class PauliGateImpl;
class PauliRotationGateImpl;

template <GateImpl T>
class GatePtr;
}  // namespace internal
using Gate = internal::GatePtr<internal::GateBase>;

enum class GateType {
    Unknown,
    I,
    GlobalPhase,
    X,
    Y,
    Z,
    H,
    S,
    Sdag,
    T,
    Tdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    P0,
    P1,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    OneTargetMatrix,
    CX,
    CZ,
    CCX,
    Swap,
    TwoTargetMatrix,
    Pauli,
    PauliRotation
};

template <internal::GateImpl T>
constexpr GateType get_gate_type() {
    if constexpr (std::is_same_v<T, internal::GateBase>) return GateType::Unknown;
    if constexpr (std::is_same_v<T, internal::IGateImpl>) return GateType::I;
    if constexpr (std::is_same_v<T, internal::GlobalPhaseGateImpl>) return GateType::GlobalPhase;
    if constexpr (std::is_same_v<T, internal::XGateImpl>) return GateType::X;
    if constexpr (std::is_same_v<T, internal::YGateImpl>) return GateType::Y;
    if constexpr (std::is_same_v<T, internal::ZGateImpl>) return GateType::Z;
    if constexpr (std::is_same_v<T, internal::HGateImpl>) return GateType::H;
    if constexpr (std::is_same_v<T, internal::SGateImpl>) return GateType::S;
    if constexpr (std::is_same_v<T, internal::SdagGateImpl>) return GateType::Sdag;
    if constexpr (std::is_same_v<T, internal::TGateImpl>) return GateType::T;
    if constexpr (std::is_same_v<T, internal::TdagGateImpl>) return GateType::Tdag;
    if constexpr (std::is_same_v<T, internal::SqrtXGateImpl>) return GateType::SqrtX;
    if constexpr (std::is_same_v<T, internal::SqrtXdagGateImpl>) return GateType::SqrtXdag;
    if constexpr (std::is_same_v<T, internal::SqrtYGateImpl>) return GateType::SqrtY;
    if constexpr (std::is_same_v<T, internal::SqrtYdagGateImpl>) return GateType::SqrtYdag;
    if constexpr (std::is_same_v<T, internal::P0GateImpl>) return GateType::P0;
    if constexpr (std::is_same_v<T, internal::P1GateImpl>) return GateType::P1;
    if constexpr (std::is_same_v<T, internal::RXGateImpl>) return GateType::RX;
    if constexpr (std::is_same_v<T, internal::RYGateImpl>) return GateType::RY;
    if constexpr (std::is_same_v<T, internal::RZGateImpl>) return GateType::RZ;
    if constexpr (std::is_same_v<T, internal::U1GateImpl>) return GateType::U1;
    if constexpr (std::is_same_v<T, internal::U2GateImpl>) return GateType::U2;
    if constexpr (std::is_same_v<T, internal::U3GateImpl>) return GateType::U3;
    if constexpr (std::is_same_v<T, internal::OneTargetMatrixGateImpl>)
        return GateType::OneTargetMatrix;
    if constexpr (std::is_same_v<T, internal::CXGateImpl>) return GateType::CX;
    if constexpr (std::is_same_v<T, internal::CZGateImpl>) return GateType::CZ;
    if constexpr (std::is_same_v<T, internal::CZGateImpl>) return GateType::CCX;
    if constexpr (std::is_same_v<T, internal::SwapGateImpl>) return GateType::Swap;
    if constexpr (std::is_same_v<T, internal::TwoTargetMatrixGateImpl>)
        return GateType::TwoTargetMatrix;
    if constexpr (std::is_same_v<T, internal::PauliGateImpl>) return GateType::Pauli;
    if constexpr (std::is_same_v<T, internal::PauliRotationGateImpl>)
        return GateType::PauliRotation;
    static_assert("unknown GateImpl");
    return GateType::Unknown;
}

namespace internal {
class GateBase : public std::enable_shared_from_this<GateBase> {
protected:
    UINT _target_mask, _control_mask;
    void check_qubit_mask_within_bounds(StateVector& state_vector) const {
        UINT full_mask = (1ULL << state_vector.n_qubits()) - 1;
        if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: Gate::update_quantum_state(StateVector& state): "
                "Target/Control qubit exceeds the number of qubits in the system.");
        }
    }

public:
    GateBase(UINT target_mask, UINT control_mask)
        : _target_mask(target_mask), _control_mask(control_mask) {
        if (_target_mask & _control_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: Gate::Gate(UINT target_mask, UINT control_mask) : Target and "
                "control qubits must not overlap.");
        }
    }
    virtual ~GateBase() = default;

    [[nodiscard]] std::vector<UINT> get_target_qubit_list() const {
        return mask_to_vector(_target_mask);
    }
    [[nodiscard]] std::vector<UINT> get_control_qubit_list() const {
        return mask_to_vector(_control_mask);
    }
    [[nodiscard]] std::vector<UINT> get_operand_qubit_list() const {
        return mask_to_vector(_target_mask | _control_mask);
    }
    [[nodiscard]] UINT get_target_qubit_mask() const { return _target_mask; }
    [[nodiscard]] UINT get_control_qubit_mask() const { return _control_mask; }
    [[nodiscard]] UINT get_operand_qubit_mask() const { return _target_mask | _control_mask; }

    [[nodiscard]] virtual Gate get_inverse() const = 0;
    [[nodiscard]] virtual std::optional<ComplexMatrix> get_matrix() const = 0;

    virtual void update_quantum_state(StateVector& state_vector) const = 0;
};

template <GateImpl T>
class GatePtr {
    friend class GateFactory;
    template <GateImpl U>
    friend class GatePtr;

private:
    std::shared_ptr<const T> _gate_ptr;
    GateType _gate_type;

public:
    GatePtr() : _gate_ptr(nullptr), _gate_type(get_gate_type<T>()) {}
    GatePtr(const GatePtr& gate) = default;
    template <GateImpl U>
    GatePtr(const std::shared_ptr<const U>& gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _gate_type = get_gate_type<T>();
            _gate_ptr = gate_ptr;
        } else if constexpr (std::is_same_v<T, internal::GateBase>) {
            // upcast
            _gate_type = get_gate_type<U>();
            _gate_ptr = std::static_pointer_cast<const T>(gate_ptr);
        } else {
            // downcast
            _gate_type = get_gate_type<T>();
            if (!(_gate_ptr = std::dynamic_pointer_cast<const T>(gate_ptr))) {
                throw std::runtime_error("invalid gate cast");
            }
        }
    }
    template <GateImpl U>
    GatePtr(const GatePtr<U>& gate) {
        if constexpr (std::is_same_v<T, U>) {
            _gate_type = gate._gate_type;
            _gate_ptr = gate._gate_ptr;
        } else if constexpr (std::is_same_v<T, internal::GateBase>) {
            // upcast
            _gate_type = gate._gate_type;
            _gate_ptr = std::static_pointer_cast<const T>(gate._gate_ptr);
        } else {
            // downcast
            if (gate._gate_type != get_gate_type<T>()) {
                throw std::runtime_error("invalid gate cast");
            }
            _gate_type = gate._gate_type;
            _gate_ptr = std::static_pointer_cast<const T>(gate._gate_ptr);
        }
    }

    GateType gate_type() const { return _gate_type; }

    const T* operator->() const {
        if (!_gate_ptr) {
            throw std::runtime_error("GatePtr::operator->(): Gate is Null");
        }
        return _gate_ptr.get();
    }
};
}  // namespace internal

}  // namespace scaluq
