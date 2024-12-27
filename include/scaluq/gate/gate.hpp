#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"

namespace scaluq {
namespace internal {
// forward declarations

template <std::floating_point Fp>
class GateBase;

template <std::floating_point Fp>
class IGateImpl;
template <std::floating_point Fp>
class GlobalPhaseGateImpl;
template <std::floating_point Fp>
class XGateImpl;
template <std::floating_point Fp>
class YGateImpl;
template <std::floating_point Fp>
class ZGateImpl;
template <std::floating_point Fp>
class HGateImpl;
template <std::floating_point Fp>
class SGateImpl;
template <std::floating_point Fp>
class SdagGateImpl;
template <std::floating_point Fp>
class TGateImpl;
template <std::floating_point Fp>
class TdagGateImpl;
template <std::floating_point Fp>
class SqrtXGateImpl;
template <std::floating_point Fp>
class SqrtXdagGateImpl;
template <std::floating_point Fp>
class SqrtYGateImpl;
template <std::floating_point Fp>
class SqrtYdagGateImpl;
template <std::floating_point Fp>
class P0GateImpl;
template <std::floating_point Fp>
class P1GateImpl;
template <std::floating_point Fp>
class RXGateImpl;
template <std::floating_point Fp>
class RYGateImpl;
template <std::floating_point Fp>
class RZGateImpl;
template <std::floating_point Fp>
class U1GateImpl;
template <std::floating_point Fp>
class U2GateImpl;
template <std::floating_point Fp>
class U3GateImpl;
template <std::floating_point Fp>
class SwapGateImpl;
template <std::floating_point Fp>
class PauliGateImpl;
template <std::floating_point Fp>
class PauliRotationGateImpl;
template <std::floating_point Fp>
class ProbablisticGateImpl;
template <std::floating_point Fp>
class SparseMatrixGateImpl;
template <std::floating_point Fp>
class DenseMatrixGateImpl;

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
    Swap,
    Pauli,
    PauliRotation,
    SparseMatrix,
    DenseMatrix,
    Probablistic
};

template <typename T, std::floating_point S>
constexpr GateType get_gate_type() {
    using TWithoutConst = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<TWithoutConst, internal::GateBase<S>>)
        return GateType::Unknown;
    else if constexpr (std::is_same_v<TWithoutConst, internal::IGateImpl<S>>)
        return GateType::I;
    else if constexpr (std::is_same_v<TWithoutConst, internal::GlobalPhaseGateImpl<S>>)
        return GateType::GlobalPhase;
    else if constexpr (std::is_same_v<TWithoutConst, internal::XGateImpl<S>>)
        return GateType::X;
    else if constexpr (std::is_same_v<TWithoutConst, internal::YGateImpl<S>>)
        return GateType::Y;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ZGateImpl<S>>)
        return GateType::Z;
    else if constexpr (std::is_same_v<TWithoutConst, internal::HGateImpl<S>>)
        return GateType::H;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SGateImpl<S>>)
        return GateType::S;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SdagGateImpl<S>>)
        return GateType::Sdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::TGateImpl<S>>)
        return GateType::T;
    else if constexpr (std::is_same_v<TWithoutConst, internal::TdagGateImpl<S>>)
        return GateType::Tdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtXGateImpl<S>>)
        return GateType::SqrtX;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtXdagGateImpl<S>>)
        return GateType::SqrtXdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtYGateImpl<S>>)
        return GateType::SqrtY;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtYdagGateImpl<S>>)
        return GateType::SqrtYdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::P0GateImpl<S>>)
        return GateType::P0;
    else if constexpr (std::is_same_v<TWithoutConst, internal::P1GateImpl<S>>)
        return GateType::P1;
    else if constexpr (std::is_same_v<TWithoutConst, internal::RXGateImpl<S>>)
        return GateType::RX;
    else if constexpr (std::is_same_v<TWithoutConst, internal::RYGateImpl<S>>)
        return GateType::RY;
    else if constexpr (std::is_same_v<TWithoutConst, internal::RZGateImpl<S>>)
        return GateType::RZ;
    else if constexpr (std::is_same_v<TWithoutConst, internal::U1GateImpl<S>>)
        return GateType::U1;
    else if constexpr (std::is_same_v<TWithoutConst, internal::U2GateImpl<S>>)
        return GateType::U2;
    else if constexpr (std::is_same_v<TWithoutConst, internal::U3GateImpl<S>>)
        return GateType::U3;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SwapGateImpl<S>>)
        return GateType::Swap;
    else if constexpr (std::is_same_v<TWithoutConst, internal::PauliGateImpl<S>>)
        return GateType::Pauli;
    else if constexpr (std::is_same_v<TWithoutConst, internal::PauliRotationGateImpl<S>>)
        return GateType::PauliRotation;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SparseMatrixGateImpl<S>>)
        return GateType::SparseMatrix;
    else if constexpr (std::is_same_v<TWithoutConst, internal::DenseMatrixGateImpl<S>>)
        return GateType::DenseMatrix;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ProbablisticGateImpl<S>>)
        return GateType::Probablistic;
    else
        static_assert(internal::lazy_false_v<T>, "unknown GateImpl");
}

namespace internal {
// GateBase テンプレートクラス
template <std::floating_point _FloatType>
class GateBase : public std::enable_shared_from_this<GateBase<_FloatType>> {
public:
    using Fp = _FloatType;

protected:
    std::uint64_t _target_mask, _control_mask;

    void check_qubit_mask_within_bounds(const StateVector<Fp>& state_vector) const;
    void check_qubit_mask_within_bounds(const StateVectorBatched<Fp>& states) const;

    std::string get_qubit_info_as_string(const std::string& indent) const;

public:
    GateBase(std::uint64_t target_mask, std::uint64_t control_mask);
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

    [[nodiscard]] virtual std::shared_ptr<const GateBase<Fp>> get_inverse() const = 0;
    [[nodiscard]] virtual internal::ComplexMatrix<Fp> get_matrix() const = 0;

    virtual void update_quantum_state(StateVector<Fp>& state_vector) const = 0;
    virtual void update_quantum_state(StateVectorBatched<Fp>& states) const = 0;

    [[nodiscard]] virtual std::string to_string(const std::string& indent = "") const = 0;

    virtual void get_as_json(Json& j) const { j = Json{{"type", "Unknown"}}; }
};

template <typename T>
concept GateImpl = std::derived_from<T, GateBase<typename T::Fp>>;

template <GateImpl T>
inline std::shared_ptr<const T> get_from_json(const Json&);

template <GateImpl T>
class GatePtr {
    friend class GateFactory;
    template <GateImpl U>
    friend class GatePtr;

private:
    std::shared_ptr<const T> _gate_ptr;
    GateType _gate_type;

public:
    using Fp = typename T::Fp;
    GatePtr() : _gate_ptr(nullptr), _gate_type(get_gate_type<T, Fp>()) {}
    template <GateImpl U>
    GatePtr(const std::shared_ptr<const U>& gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _gate_type = get_gate_type<T, Fp>();
            _gate_ptr = gate_ptr;
        } else if constexpr (std::is_same_v<T, internal::GateBase<Fp>>) {
            // upcast
            _gate_type = get_gate_type<U, Fp>();
            _gate_ptr = std::static_pointer_cast<const T>(gate_ptr);
        } else {
            // downcast
            _gate_type = get_gate_type<T, Fp>();
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
        } else if constexpr (std::is_same_v<T, internal::GateBase<Fp>>) {
            // upcast
            _gate_type = gate._gate_type;
            _gate_ptr = std::static_pointer_cast<const T>(gate._gate_ptr);
        } else {
            // downcast
            if (gate._gate_type != get_gate_type<T, Fp>()) {
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

    friend void to_json(Json& j, const GatePtr& gate) { gate->get_as_json(j); }

    friend void from_json(const Json& j, GatePtr& gate) {
        std::string type = j.at("type");

        // clang-format off
        if (type == "I") gate = get_from_json<IGateImpl<Fp>>(j);
        else if (type == "GlobalPhase") gate = get_from_json<GlobalPhaseGateImpl<Fp>>(j);
        else if (type == "X") gate = get_from_json<XGateImpl<Fp>>(j);
        else if (type == "Y") gate = get_from_json<YGateImpl<Fp>>(j);
        else if (type == "Z") gate = get_from_json<ZGateImpl<Fp>>(j);
        else if (type == "H") gate = get_from_json<HGateImpl<Fp>>(j);
        else if (type == "S") gate = get_from_json<SGateImpl<Fp>>(j);
        else if (type == "Sdag") gate = get_from_json<SdagGateImpl<Fp>>(j);
        else if (type == "T") gate = get_from_json<TGateImpl<Fp>>(j);
        else if (type == "Tdag") gate = get_from_json<TdagGateImpl<Fp>>(j);
        else if (type == "SqrtX") gate = get_from_json<SqrtXGateImpl<Fp>>(j);
        else if (type == "SqrtXdag") gate = get_from_json<SqrtXdagGateImpl<Fp>>(j);
        else if (type == "SqrtY") gate = get_from_json<SqrtYGateImpl<Fp>>(j);
        else if (type == "SqrtYdag") gate = get_from_json<SqrtYdagGateImpl<Fp>>(j);
        else if (type == "RX") gate = get_from_json<RXGateImpl<Fp>>(j);
        else if (type == "RY") gate = get_from_json<RYGateImpl<Fp>>(j);
        else if (type == "RZ") gate = get_from_json<RZGateImpl<Fp>>(j);
        else if (type == "U1") gate = get_from_json<U1GateImpl<Fp>>(j);
        else if (type == "U2") gate = get_from_json<U2GateImpl<Fp>>(j);
        else if (type == "U3") gate = get_from_json<U3GateImpl<Fp>>(j);
        else if (type == "Swap") gate = get_from_json<SwapGateImpl<Fp>>(j);
        else if (type == "Pauli") gate = get_from_json<PauliGateImpl<Fp>>(j);
        else if (type == "PauliRotation") gate = get_from_json<PauliRotationGateImpl<Fp>>(j);
        else if (type == "Probablistic") gate = get_from_json<ProbablisticGateImpl<Fp>>(j);
        // clang-format on
    }
};

}  // namespace internal

template <std::floating_point Fp>
using Gate = internal::GatePtr<internal::GateBase<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
#define DEF_GATE_BASE(GATE_TYPE, FLOAT, DESCRIPTION)                                         \
    nb::class_<GATE_TYPE<FLOAT>>(m, #GATE_TYPE, DESCRIPTION)                                 \
        .def("gate_type", &GATE_TYPE<FLOAT>::gate_type, "Get gate type as `GateType` enum.") \
        .def(                                                                                \
            "target_qubit_list",                                                             \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->target_qubit_list(); },          \
            "Get target qubits as `list[int]`. **Control qubits is not included.**")         \
        .def(                                                                                \
            "control_qubit_list",                                                            \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->control_qubit_list(); },         \
            "Get control qubits as `list[int]`.")                                            \
        .def(                                                                                \
            "operand_qubit_list",                                                            \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->operand_qubit_list(); },         \
            "Get target and control qubits as `list[int]`.")                                 \
        .def(                                                                                \
            "target_qubit_mask",                                                             \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->target_qubit_mask(); },          \
            "Get target qubits as mask. **Control qubits is not included.**")                \
        .def(                                                                                \
            "control_qubit_mask",                                                            \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->control_qubit_mask(); },         \
            "Get control qubits as mask.")                                                   \
        .def(                                                                                \
            "operand_qubit_mask",                                                            \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->operand_qubit_mask(); },         \
            "Get target and control qubits as mask.")                                        \
        .def(                                                                                \
            "get_inverse",                                                                   \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->get_inverse(); },                \
            "Generate inverse gate as `Gate` type. If not exists, return None.")             \
        .def(                                                                                \
            "update_quantum_state",                                                          \
            [](const GATE_TYPE<FLOAT>& gate, StateVector<FLOAT>& state_vector) {             \
                gate->update_quantum_state(state_vector);                                    \
            },                                                                               \
            "Apply gate to `state_vector`. `state_vector` in args is directly updated.")     \
        .def(                                                                                \
            "get_matrix",                                                                    \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->get_matrix(); },                 \
            "Get matrix representation of the gate.")                                        \
        .def(                                                                                \
            "to_string",                                                                     \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->to_string(""); },                \
            "Get string representation of the gate.")                                        \
        .def(                                                                                \
            "__str__",                                                                       \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->to_string(""); },                \
            "Get string representation of the gate.")                                        \
        .def(                                                                                \
            "to_json",                                                                       \
            [](const GATE_TYPE<FLOAT>& gate) { return Json(gate).dump(); },                  \
            "Get JSON representation of the gate.")                                          \
        .def(                                                                                \
            "load_json",                                                                     \
            [](GATE_TYPE<FLOAT>& gate, const std::string& str) {                             \
                gate = nlohmann::json::parse(str);                                           \
            },                                                                               \
            "Read an object from the JSON representation of the gate.")

template <std::floating_point Fp>
nb::class_<Gate<Fp>> gate_base_def;

#define DEF_GATE(GATE_TYPE, FLOAT, DESCRIPTION)                                                 \
    ::scaluq::internal::gate_base_def<FLOAT>.def(nb::init<GATE_TYPE<FLOAT>>(),                  \
                                                 "Upcast from `" #GATE_TYPE "`.");              \
    DEF_GATE_BASE(                                                                              \
        GATE_TYPE,                                                                              \
        FLOAT,                                                                                  \
        DESCRIPTION                                                                             \
        "\n\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).") \
        .def(nb::init<Gate<FLOAT>>())

void bind_gate_gate_hpp_without_precision(nb::module_& m) {
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
        .value("Swap", GateType::Swap)
        .value("Pauli", GateType::Pauli)
        .value("PauliRotation", GateType::PauliRotation)
        .value("SparseMatrix", GateType::SparseMatrix)
        .value("DenseMatrix", GateType::DenseMatrix);
}

template <std::floating_point Fp>
void bind_gate_gate_hpp(nb::module_& m) {
    gate_base_def<Fp> =
        DEF_GATE_BASE(Gate,
                      Fp,
                      "General class of QuantumGate.\n\n.. note:: Downcast to requred to use "
                      "gate-specific functions.")
            .def(nb::init<Gate<Fp>>(), "Just copy shallowly.");
}
}  // namespace internal
#endif

}  // namespace scaluq
