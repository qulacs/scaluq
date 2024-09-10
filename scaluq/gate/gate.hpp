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
class SwapGateImpl;
class TwoTargetMatrixGateImpl;
class PauliGateImpl;
class PauliRotationGateImpl;
class ProbablisticGateImpl;

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
    Swap,
    TwoTargetMatrix,
    Pauli,
    PauliRotation,
    Probablistic,
    Error
};

template <internal::GateImpl T>
constexpr GateType get_gate_type() {
    using TWithoutConst = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<TWithoutConst, internal::GateBase>) return GateType::Unknown;
    if constexpr (std::is_same_v<TWithoutConst, internal::IGateImpl>) return GateType::I;
    if constexpr (std::is_same_v<TWithoutConst, internal::GlobalPhaseGateImpl>)
        return GateType::GlobalPhase;
    if constexpr (std::is_same_v<TWithoutConst, internal::XGateImpl>) return GateType::X;
    if constexpr (std::is_same_v<TWithoutConst, internal::YGateImpl>) return GateType::Y;
    if constexpr (std::is_same_v<TWithoutConst, internal::ZGateImpl>) return GateType::Z;
    if constexpr (std::is_same_v<TWithoutConst, internal::HGateImpl>) return GateType::H;
    if constexpr (std::is_same_v<TWithoutConst, internal::SGateImpl>) return GateType::S;
    if constexpr (std::is_same_v<TWithoutConst, internal::SdagGateImpl>) return GateType::Sdag;
    if constexpr (std::is_same_v<TWithoutConst, internal::TGateImpl>) return GateType::T;
    if constexpr (std::is_same_v<TWithoutConst, internal::TdagGateImpl>) return GateType::Tdag;
    if constexpr (std::is_same_v<TWithoutConst, internal::SqrtXGateImpl>) return GateType::SqrtX;
    if constexpr (std::is_same_v<TWithoutConst, internal::SqrtXdagGateImpl>)
        return GateType::SqrtXdag;
    if constexpr (std::is_same_v<TWithoutConst, internal::SqrtYGateImpl>) return GateType::SqrtY;
    if constexpr (std::is_same_v<TWithoutConst, internal::SqrtYdagGateImpl>)
        return GateType::SqrtYdag;
    if constexpr (std::is_same_v<TWithoutConst, internal::P0GateImpl>) return GateType::P0;
    if constexpr (std::is_same_v<TWithoutConst, internal::P1GateImpl>) return GateType::P1;
    if constexpr (std::is_same_v<TWithoutConst, internal::RXGateImpl>) return GateType::RX;
    if constexpr (std::is_same_v<TWithoutConst, internal::RYGateImpl>) return GateType::RY;
    if constexpr (std::is_same_v<TWithoutConst, internal::RZGateImpl>) return GateType::RZ;
    if constexpr (std::is_same_v<TWithoutConst, internal::U1GateImpl>) return GateType::U1;
    if constexpr (std::is_same_v<TWithoutConst, internal::U2GateImpl>) return GateType::U2;
    if constexpr (std::is_same_v<TWithoutConst, internal::U3GateImpl>) return GateType::U3;
    if constexpr (std::is_same_v<TWithoutConst, internal::OneTargetMatrixGateImpl>)
        return GateType::OneTargetMatrix;
    if constexpr (std::is_same_v<TWithoutConst, internal::SwapGateImpl>) return GateType::Swap;
    if constexpr (std::is_same_v<TWithoutConst, internal::TwoTargetMatrixGateImpl>)
        return GateType::TwoTargetMatrix;
    if constexpr (std::is_same_v<TWithoutConst, internal::PauliGateImpl>) return GateType::Pauli;
    if constexpr (std::is_same_v<TWithoutConst, internal::PauliRotationGateImpl>)
        return GateType::PauliRotation;
    if constexpr (std::is_same_v<TWithoutConst, internal::ProbablisticGateImpl>)
        return GateType::Probablistic;
    return GateType::Error;
}

namespace internal {
class GateBase : public std::enable_shared_from_this<GateBase> {
protected:
    std::uint64_t _target_mask, _control_mask;
    void check_qubit_mask_within_bounds(const StateVector& state_vector) const {
        std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
        if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: Gate::update_quantum_state(StateVector& state): "
                "Target/Control qubit exceeds the number of qubits in the system.");
        }
    }

    std::string get_qubit_info_as_string(const std::string& indent) const {
        std::ostringstream ss;
        auto targets = target_qubit_list();
        auto controls = control_qubit_list();
        ss << indent << "  Target Qubits: {";
        for (std::uint32_t i = 0; i < targets.size(); ++i)
            ss << targets[i] << (i == targets.size() - 1 ? "" : ", ");
        ss << "}\n";
        ss << indent << "  Control Qubits: {";
        for (std::uint32_t i = 0; i < controls.size(); ++i)
            ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
        ss << "}";
        return ss.str();
    }

public:
    GateBase(std::uint64_t target_mask, std::uint64_t control_mask)
        : _target_mask(target_mask), _control_mask(control_mask) {
        if (_target_mask & _control_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: Gate::Gate(std::uint64_t target_mask, std::uint64_t control_mask) : Target "
                "and control qubits must not overlap.");
        }
    }
    virtual ~GateBase() = default;

    [[nodiscard]] virtual std::vector<std::uint64_t> target_qubit_list() const {
        return mask_to_vector(_target_mask);
    }
    [[nodiscard]] virtual std::vector<std::uint64_t> control_qubit_list() const {
        return mask_to_vector(_control_mask);
    }
    [[nodiscard]] virtual std::vector<std::uint64_t> operand_qubit_list() const {
        return mask_to_vector(_target_mask | _control_mask);
    }
    [[nodiscard]] virtual std::uint64_t target_qubit_mask() const { return _target_mask; }
    [[nodiscard]] virtual std::uint64_t control_qubit_mask() const { return _control_mask; }
    [[nodiscard]] virtual std::uint64_t operand_qubit_mask() const {
        return _target_mask | _control_mask;
    }

    [[nodiscard]] virtual Gate get_inverse() const = 0;
    [[nodiscard]] virtual internal::ComplexMatrix get_matrix() const = 0;

    virtual void update_quantum_state(StateVector& state_vector) const = 0;

    [[nodiscard]] virtual std::string to_string(const std::string& indent = "") const = 0;
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
            if ((_gate_type = get_gate_type<T>()) == GateType::Error) {
                throw std::runtime_error("Unknown GateType");
            }
            _gate_ptr = gate_ptr;
        } else if constexpr (std::is_same_v<T, internal::GateBase>) {
            // upcast
            if ((_gate_type = get_gate_type<U>()) == GateType::Error) {
                throw std::runtime_error("Unknown GateType");
            }
            _gate_ptr = std::static_pointer_cast<const T>(gate_ptr);
        } else {
            // downcast
            if ((_gate_type = get_gate_type<T>()) == GateType::Error) {
                throw std::runtime_error("Unknown GateType");
            }
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

    friend std::ostream& operator<<(std::ostream& os, GatePtr gate) {
        os << gate->to_string();
        return os;
    }
};
}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
#define DEF_GATE_BASE(GATE_TYPE, DESCRIPTION)                                            \
    nb::class_<GATE_TYPE>(m, #GATE_TYPE, DESCRIPTION)                                    \
        .def("gate_type", &GATE_TYPE::gate_type, "Get gate type as `GateType` enum.")    \
        .def(                                                                            \
            "target_qubit_list",                                                         \
            [](const GATE_TYPE& gate) { return gate->target_qubit_list(); },             \
            "Get target qubits as `list[int]`. **Control qubits is not included.**")     \
        .def(                                                                            \
            "control_qubit_list",                                                        \
            [](const GATE_TYPE& gate) { return gate->control_qubit_list(); },            \
            "Get control qubits as `list[int]`.")                                        \
        .def(                                                                            \
            "operand_qubit_list",                                                        \
            [](const GATE_TYPE& gate) { return gate->operand_qubit_list(); },            \
            "Get target and control qubits as `list[int]`.")                             \
        .def(                                                                            \
            "target_qubit_mask",                                                         \
            [](const GATE_TYPE& gate) { return gate->target_qubit_mask(); },             \
            "Get target qubits as mask. **Control qubits is not included.**")            \
        .def(                                                                            \
            "control_qubit_mask",                                                        \
            [](const GATE_TYPE& gate) { return gate->control_qubit_mask(); },            \
            "Get control qubits as mask.")                                               \
        .def(                                                                            \
            "operand_qubit_mask",                                                        \
            [](const GATE_TYPE& gate) { return gate->operand_qubit_mask(); },            \
            "Get target and control qubits as mask.")                                    \
        .def(                                                                            \
            "get_inverse",                                                               \
            [](const GATE_TYPE& gate) { return gate->get_inverse(); },                   \
            "Generate inverse gate as `Gate` type. If not exists, return None.")         \
        .def(                                                                            \
            "update_quantum_state",                                                      \
            [](const GATE_TYPE& gate, StateVector& state_vector) {                       \
                gate->update_quantum_state(state_vector);                                \
            },                                                                           \
            "Apply gate to `state_vector`. `state_vector` in args is directly updated.") \
        .def(                                                                            \
            "get_matrix",                                                                \
            [](const GATE_TYPE& gate) { return gate->get_matrix(); },                    \
            "Get matrix representation of the gate.")

nb::class_<Gate> gate_base_def;

#define DEF_GATE(GATE_TYPE, DESCRIPTION)                                                           \
    ::scaluq::internal::gate_base_def.def(nb::init<GATE_TYPE>(), "Upcast from `" #GATE_TYPE "`."); \
    DEF_GATE_BASE(                                                                                 \
        GATE_TYPE,                                                                                 \
        DESCRIPTION                                                                                \
        "\n\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).")    \
        .def(nb::init<Gate>())

void bind_gate_gate_hpp(nb::module_& m) {
    nb::enum_<GateType>(m, "GateType", "Enum of Gate Type.")
        .value("I", GateType::I)
        .value("GlobalPhase", GateType::GlobalPhase)
        .value("X", GateType::X)
        .value("Y", GateType::Y)
        .value("Z", GateType::Z)
        .value("H", GateType::H)
        .value("S", GateType::S)
        .value("Sdag", GateType::Sdag)
        .value("T", GateType::T)
        .value("Tdag", GateType::Tdag)
        .value("SqrtX", GateType::SqrtX)
        .value("SqrtXdag", GateType::SqrtXdag)
        .value("SqrtY", GateType::SqrtY)
        .value("SqrtYdag", GateType::SqrtYdag)
        .value("P0", GateType::P0)
        .value("P1", GateType::P1)
        .value("RX", GateType::RX)
        .value("RY", GateType::RY)
        .value("RZ", GateType::RZ)
        .value("U1", GateType::U1)
        .value("U2", GateType::U2)
        .value("U3", GateType::U3)
        .value("OneTargetMatrix", GateType::OneTargetMatrix)
        .value("Swap", GateType::Swap)
        .value("TwoTargetMatrix", GateType::TwoTargetMatrix)
        .value("Pauli", GateType::Pauli)
        .value("PauliRotation", GateType::PauliRotation);

    gate_base_def =
        DEF_GATE_BASE(Gate,
                      "General class of QuantumGate.\n\n.. note:: Downcast to requred to use "
                      "gate-specific functions.")
            .def(nb::init<Gate>(), "Just copy shallowly.");
}
}  // namespace internal
#endif

}  // namespace scaluq
