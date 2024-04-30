#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"
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
class OneQubitMatrixGateImpl;
class CXGateImpl;
class CZGateImpl;
class SwapGateImpl;
class TwoQubitMatrixGateImpl;
class FusedSwapGateImpl;
class PauliGateImpl;
class PauliRotationGateImpl;
class PRXGateImpl;
class PRYGateImpl;
class PRZGateImpl;
class PPauliRotationGateImpl;

template <GateImpl T>
class GatePtr;
}  // namespace internal

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
    OneQubitMatrix,
    CX,
    CZ,
    Swap,
    TwoQubitMatrix,
    FusedSwap,
    Pauli,
    PauliRotation,
    PRX,
    PRY,
    PRZ,
    PPauliRotation
};

using Gate = internal::GatePtr<internal::GateBase>;

namespace internal {
class GateBase {
public:
    virtual ~GateBase() = default;

    [[nodiscard]] virtual std::vector<UINT> get_target_qubit_list() const = 0;
    [[nodiscard]] virtual std::vector<UINT> get_control_qubit_list() const = 0;

    [[nodiscard]] virtual Gate copy() const = 0;
    [[nodiscard]] virtual Gate get_inverse() const = 0;
    [[nodiscard]] virtual std::optional<ComplexMatrix> get_matrix() const = 0;

    virtual void update_quantum_state(StateVector& state_vector) const = 0;
};

class ParametricGateBase : public GateBase {
public:
    void update_quantum_state([[maybe_unused]] StateVector& state_vector) const override {
        throw std::runtime_error(
            "ParametricGateBase::update_quantum_state(): ParametricGate cannot run "
            "update_quantum_state without parameters. Please cast to ParametricGate type or more "
            "specific class and run update_quantum_state(StateVector&, double).");
    }
    [[nodiscard]] virtual Gate get_inverse() const;
    [[nodiscard]] virtual std::optional<ComplexMatrix> get_matrix() const { return std::nullopt; }
    virtual void update_quantum_state(StateVector& state_vector, double parameter) const = 0;
};

template <GateImpl T>
constexpr GateType get_gate_type() {
    if constexpr (std::is_same_v<T, GateBase>) return GateType::Unknown;
    if constexpr (std::is_same_v<T, ParametricGateBase>) return GateType::Unknown;
    if constexpr (std::is_same_v<T, IGateImpl>) return GateType::I;
    if constexpr (std::is_same_v<T, GlobalPhaseGateImpl>) return GateType::GlobalPhase;
    if constexpr (std::is_same_v<T, XGateImpl>) return GateType::X;
    if constexpr (std::is_same_v<T, YGateImpl>) return GateType::Y;
    if constexpr (std::is_same_v<T, ZGateImpl>) return GateType::Z;
    if constexpr (std::is_same_v<T, HGateImpl>) return GateType::H;
    if constexpr (std::is_same_v<T, SGateImpl>) return GateType::S;
    if constexpr (std::is_same_v<T, SdagGateImpl>) return GateType::Sdag;
    if constexpr (std::is_same_v<T, TGateImpl>) return GateType::T;
    if constexpr (std::is_same_v<T, TdagGateImpl>) return GateType::Tdag;
    if constexpr (std::is_same_v<T, SqrtXGateImpl>) return GateType::SqrtX;
    if constexpr (std::is_same_v<T, SqrtXdagGateImpl>) return GateType::SqrtXdag;
    if constexpr (std::is_same_v<T, SqrtYGateImpl>) return GateType::SqrtY;
    if constexpr (std::is_same_v<T, SqrtYdagGateImpl>) return GateType::SqrtYdag;
    if constexpr (std::is_same_v<T, P0GateImpl>) return GateType::P0;
    if constexpr (std::is_same_v<T, P1GateImpl>) return GateType::P1;
    if constexpr (std::is_same_v<T, RXGateImpl>) return GateType::RX;
    if constexpr (std::is_same_v<T, RYGateImpl>) return GateType::RY;
    if constexpr (std::is_same_v<T, RZGateImpl>) return GateType::RZ;
    if constexpr (std::is_same_v<T, U1GateImpl>) return GateType::U1;
    if constexpr (std::is_same_v<T, U2GateImpl>) return GateType::U2;
    if constexpr (std::is_same_v<T, U3GateImpl>) return GateType::U3;
    if constexpr (std::is_same_v<T, OneQubitMatrixGateImpl>) return GateType::OneQubitMatrix;
    if constexpr (std::is_same_v<T, CXGateImpl>) return GateType::CX;
    if constexpr (std::is_same_v<T, CZGateImpl>) return GateType::CZ;
    if constexpr (std::is_same_v<T, SwapGateImpl>) return GateType::Swap;
    if constexpr (std::is_same_v<T, TwoQubitMatrixGateImpl>) return GateType::TwoQubitMatrix;
    if constexpr (std::is_same_v<T, FusedSwapGateImpl>) return GateType::FusedSwap;
    if constexpr (std::is_same_v<T, PauliGateImpl>) return GateType::Pauli;
    if constexpr (std::is_same_v<T, PauliRotationGateImpl>) return GateType::PauliRotation;
    if constexpr (std::is_same_v<T, PRXGateImpl>) return GateType::PRX;
    if constexpr (std::is_same_v<T, PRYGateImpl>) return GateType::PRY;
    if constexpr (std::is_same_v<T, PRZGateImpl>) return GateType::PRZ;
    if constexpr (std::is_same_v<T, PPauliRotationGateImpl>) return GateType::PPauliRotation;
    static_assert("unknown GateImpl");
    return GateType::Unknown;
}

template <GateImpl T>
class GatePtr {
    friend class GateFactory;
    template <GateImpl U>
    friend class GatePtr;

private:
    std::shared_ptr<T> _gate_ptr;
    GateType _gate_type;

public:
    GatePtr() : _gate_ptr(nullptr), _gate_type(get_gate_type<T>()) {}
    GatePtr(const GatePtr& gate) = default;
    template <GateImpl U>
    GatePtr(const std::shared_ptr<U>& gate_ptr) {
        _gate_type = std::max(get_gate_type<T>(), get_gate_type<U>());
        if (!(_gate_ptr = std::dynamic_pointer_cast<T>(gate_ptr))) {
            throw std::runtime_error("invalid gate cast");
        }
    }
    template <GateImpl U>
    GatePtr(const GatePtr<U>& gate) {
        _gate_type = gate._gate_type;
        if (!(_gate_ptr = std::dynamic_pointer_cast<T>(gate._gate_ptr))) {
            throw std::runtime_error("invalid gate cast");
        }
    }

    GateType gate_type() const { return _gate_type; }

    bool is_parametric() const { return _gate_type >= GateType::PRX; }

    GatePtr<ParametricGateBase> to_parametric_gate() const {
        if (!is_parametric()) {
            throw std::runtime_error("GatePtr::to_parametric_gate(): Gate is not parametric");
        }
        return GatePtr<ParametricGateBase>(*this);
    }

    T* operator->() const {
        if (!_gate_ptr) {
            throw std::runtime_error("GatePtr::operator->(): Gate is Null");
        }
        return _gate_ptr.get();
    }
};

Gate ParametricGateBase::get_inverse() const {
    throw std::runtime_error(
        "ParametricGateBase::get_inverse(): ParametricGate does not support get_inverse().");
}
}  // namespace internal

using ParametricGate = internal::GatePtr<internal::ParametricGateBase>;

}  // namespace scaluq
