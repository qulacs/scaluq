#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"

namespace scaluq {
namespace internal {
// forward declarations

template <Precision Prec, ExecutionSpace Space>
class GateBase;

template <Precision Prec, ExecutionSpace Space>
class IGateImpl;
template <Precision Prec, ExecutionSpace Space>
class GlobalPhaseGateImpl;
template <Precision Prec, ExecutionSpace Space>
class XGateImpl;
template <Precision Prec, ExecutionSpace Space>
class YGateImpl;
template <Precision Prec, ExecutionSpace Space>
class ZGateImpl;
template <Precision Prec, ExecutionSpace Space>
class HGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SdagGateImpl;
template <Precision Prec, ExecutionSpace Space>
class TGateImpl;
template <Precision Prec, ExecutionSpace Space>
class TdagGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtXGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtXdagGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtYGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtYdagGateImpl;
template <Precision Prec, ExecutionSpace Space>
class P0GateImpl;
template <Precision Prec, ExecutionSpace Space>
class P1GateImpl;
template <Precision Prec, ExecutionSpace Space>
class RXGateImpl;
template <Precision Prec, ExecutionSpace Space>
class RYGateImpl;
template <Precision Prec, ExecutionSpace Space>
class RZGateImpl;
template <Precision Prec, ExecutionSpace Space>
class U1GateImpl;
template <Precision Prec, ExecutionSpace Space>
class U2GateImpl;
template <Precision Prec, ExecutionSpace Space>
class U3GateImpl;
template <Precision Prec, ExecutionSpace Space>
class SwapGateImpl;
template <Precision Prec, ExecutionSpace Space>
class PauliGateImpl;
template <Precision Prec, ExecutionSpace Space>
class PauliRotationGateImpl;
template <Precision Prec, ExecutionSpace Space>
class ProbabilisticGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SparseMatrixGateImpl;
template <Precision Prec, ExecutionSpace Space>
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
    Probabilistic
};

template <typename T, Precision Prec, ExecutionSpace Space>
constexpr GateType get_gate_type() {
    using TWithoutConst = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<TWithoutConst, internal::GateBase<Prec, Space>>)
        return GateType::Unknown;
    else if constexpr (std::is_same_v<TWithoutConst, internal::IGateImpl<Prec, Space>>)
        return GateType::I;
    else if constexpr (std::is_same_v<TWithoutConst, internal::GlobalPhaseGateImpl<Prec, Space>>)
        return GateType::GlobalPhase;
    else if constexpr (std::is_same_v<TWithoutConst, internal::XGateImpl<Prec, Space>>)
        return GateType::X;
    else if constexpr (std::is_same_v<TWithoutConst, internal::YGateImpl<Prec, Space>>)
        return GateType::Y;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ZGateImpl<Prec, Space>>)
        return GateType::Z;
    else if constexpr (std::is_same_v<TWithoutConst, internal::HGateImpl<Prec, Space>>)
        return GateType::H;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SGateImpl<Prec, Space>>)
        return GateType::S;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SdagGateImpl<Prec, Space>>)
        return GateType::Sdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::TGateImpl<Prec, Space>>)
        return GateType::T;
    else if constexpr (std::is_same_v<TWithoutConst, internal::TdagGateImpl<Prec, Space>>)
        return GateType::Tdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtXGateImpl<Prec, Space>>)
        return GateType::SqrtX;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtXdagGateImpl<Prec, Space>>)
        return GateType::SqrtXdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtYGateImpl<Prec, Space>>)
        return GateType::SqrtY;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SqrtYdagGateImpl<Prec, Space>>)
        return GateType::SqrtYdag;
    else if constexpr (std::is_same_v<TWithoutConst, internal::P0GateImpl<Prec, Space>>)
        return GateType::P0;
    else if constexpr (std::is_same_v<TWithoutConst, internal::P1GateImpl<Prec, Space>>)
        return GateType::P1;
    else if constexpr (std::is_same_v<TWithoutConst, internal::RXGateImpl<Prec, Space>>)
        return GateType::RX;
    else if constexpr (std::is_same_v<TWithoutConst, internal::RYGateImpl<Prec, Space>>)
        return GateType::RY;
    else if constexpr (std::is_same_v<TWithoutConst, internal::RZGateImpl<Prec, Space>>)
        return GateType::RZ;
    else if constexpr (std::is_same_v<TWithoutConst, internal::U1GateImpl<Prec, Space>>)
        return GateType::U1;
    else if constexpr (std::is_same_v<TWithoutConst, internal::U2GateImpl<Prec, Space>>)
        return GateType::U2;
    else if constexpr (std::is_same_v<TWithoutConst, internal::U3GateImpl<Prec, Space>>)
        return GateType::U3;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SwapGateImpl<Prec, Space>>)
        return GateType::Swap;
    else if constexpr (std::is_same_v<TWithoutConst, internal::PauliGateImpl<Prec, Space>>)
        return GateType::Pauli;
    else if constexpr (std::is_same_v<TWithoutConst, internal::PauliRotationGateImpl<Prec, Space>>)
        return GateType::PauliRotation;
    else if constexpr (std::is_same_v<TWithoutConst, internal::SparseMatrixGateImpl<Prec, Space>>)
        return GateType::SparseMatrix;
    else if constexpr (std::is_same_v<TWithoutConst, internal::DenseMatrixGateImpl<Prec, Space>>)
        return GateType::DenseMatrix;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ProbabilisticGateImpl<Prec, Space>>)
        return GateType::Probabilistic;
    else
        static_assert(internal::lazy_false_v<T>, "unknown GateImpl");
}

namespace internal {
// GateBase テンプレートクラス
template <Precision _Prec, ExecutionSpace _Space>
class GateBase : public std::enable_shared_from_this<GateBase<_Prec, _Space>> {
public:
    constexpr static Precision Prec = _Prec;
    constexpr static ExecutionSpace Space = _Space;
    using FloatType = Float<Prec>;
    using ComplexType = Complex<Prec>;

protected:
    std::uint64_t _target_mask, _control_mask, _control_value_mask;

    void check_qubit_mask_within_bounds(const StateVector<Prec, Space>& state_vector) const;
    void check_qubit_mask_within_bounds(const StateVectorBatched<Prec, Space>& states) const;

    std::string get_qubit_info_as_string(const std::string& indent) const;

public:
    GateBase(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask);
    virtual ~GateBase() = default;

    [[nodiscard]] virtual std::vector<std::uint64_t> target_qubit_list() const {
        return mask_to_vector(_target_mask);
    }
    [[nodiscard]] virtual std::vector<std::uint64_t> control_qubit_list() const {
        return mask_to_vector(_control_mask);
    }
    [[nodiscard]] virtual std::vector<std::uint64_t> control_value_list() const {
        return mask_to_vector(_control_mask, _control_value_mask);
    }
    [[nodiscard]] virtual std::vector<std::uint64_t> operand_qubit_list() const {
        return mask_to_vector(_target_mask | _control_mask);
    }
    [[nodiscard]] virtual std::uint64_t target_qubit_mask() const { return _target_mask; }
    [[nodiscard]] virtual std::uint64_t control_qubit_mask() const { return _control_mask; }
    [[nodiscard]] virtual std::uint64_t control_value_mask() const { return _control_value_mask; }
    [[nodiscard]] virtual std::uint64_t operand_qubit_mask() const {
        return _target_mask | _control_mask;
    }

    [[nodiscard]] virtual std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const = 0;
    [[nodiscard]] virtual ComplexMatrix get_matrix() const = 0;

    virtual void update_quantum_state(StateVector<Prec, Space>& state_vector) const = 0;
    virtual void update_quantum_state(StateVectorBatched<Prec, Space>& states) const = 0;

    [[nodiscard]] virtual std::string to_string(const std::string& indent = "") const = 0;

    virtual void get_as_json(Json& j) const { j = Json{{"type", "Unknown"}}; }
};

template <typename T>
concept GateImpl = std::derived_from<T, GateBase<T::Prec, T::Space>>;

template <GateImpl T>
struct GetGateFromJson {
    static std::shared_ptr<const T> get(const Json& j) {
        throw std::runtime_error("GetGateFromJson<T>::get() is not implemented");
    }
};
#define DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(Impl)                  \
    template <Precision Prec, ExecutionSpace Space>                         \
    struct GetGateFromJson<Impl<Prec, Space>> {                             \
        static std::shared_ptr<const Impl<Prec, Space>> get(const Json& j); \
    };
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(IGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(GlobalPhaseGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(XGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(YGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ZGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(HGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SdagGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(TGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(TdagGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SqrtXGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SqrtXdagGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SqrtYGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SqrtYdagGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(P0GateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(P1GateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(RXGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(RYGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(RZGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(U1GateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(U2GateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(U3GateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SwapGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(PauliGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(PauliRotationGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ProbabilisticGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(SparseMatrixGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(DenseMatrixGateImpl)
#undef DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION

template <GateImpl T>
class GatePtr {
    friend class GateFactory;
    template <GateImpl U>
    friend class GatePtr;

private:
    std::shared_ptr<const T> _gate_ptr;
    GateType _gate_type;

public:
    constexpr static Precision Prec = T::Prec;
    constexpr static ExecutionSpace Space = T::Space;
    using FloatType = Float<Prec>;
    using ComplexType = Complex<Prec>;
    GatePtr() : _gate_ptr(nullptr), _gate_type(get_gate_type<T, Prec, Space>()) {}
    template <GateImpl U>
    GatePtr(const std::shared_ptr<const U>& gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _gate_type = get_gate_type<T, Prec, Space>();
            _gate_ptr = gate_ptr;
        } else if constexpr (std::is_same_v<T, GateBase<Prec, Space>>) {
            // upcast
            _gate_type = get_gate_type<U, Prec, Space>();
            _gate_ptr = std::static_pointer_cast<const T>(gate_ptr);
        } else {
            // downcast
            _gate_type = get_gate_type<T, Prec, Space>();
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
        } else if constexpr (std::is_same_v<T, GateBase<Prec, Space>>) {
            // upcast
            _gate_type = gate._gate_type;
            _gate_ptr = std::static_pointer_cast<const T>(gate._gate_ptr);
        } else {
            // downcast
            if (gate._gate_type != get_gate_type<T, Prec, Space>()) {
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
        os << gate->to_string("");
        return os;
    }

    friend void to_json(Json& j, const GatePtr& gate) { gate->get_as_json(j); }

    friend void from_json(const Json& j, GatePtr& gate) {
        std::string type = j.at("type");

        // clang-format off
        if (type == "I") gate = GetGateFromJson<IGateImpl<Prec, Space>>::get(j);
        else if (type == "GlobalPhase") gate = GetGateFromJson<GlobalPhaseGateImpl<Prec, Space>>::get(j);
        else if (type == "X") gate = GetGateFromJson<XGateImpl<Prec, Space>>::get(j);
        else if (type == "Y") gate = GetGateFromJson<YGateImpl<Prec, Space>>::get(j);
        else if (type == "Z") gate = GetGateFromJson<ZGateImpl<Prec, Space>>::get(j);
        else if (type == "H") gate = GetGateFromJson<HGateImpl<Prec, Space>>::get(j);
        else if (type == "S") gate = GetGateFromJson<SGateImpl<Prec, Space>>::get(j);
        else if (type == "Sdag") gate = GetGateFromJson<SdagGateImpl<Prec, Space>>::get(j);
        else if (type == "T") gate = GetGateFromJson<TGateImpl<Prec, Space>>::get(j);
        else if (type == "Tdag") gate = GetGateFromJson<TdagGateImpl<Prec, Space>>::get(j);
        else if (type == "SqrtX") gate = GetGateFromJson<SqrtXGateImpl<Prec, Space>>::get(j);
        else if (type == "SqrtXdag") gate = GetGateFromJson<SqrtXdagGateImpl<Prec, Space>>::get(j);
        else if (type == "SqrtY") gate = GetGateFromJson<SqrtYGateImpl<Prec, Space>>::get(j);
        else if (type == "SqrtYdag") gate = GetGateFromJson<SqrtYdagGateImpl<Prec, Space>>::get(j);
        else if (type == "RX") gate = GetGateFromJson<RXGateImpl<Prec, Space>>::get(j);
        else if (type == "RY") gate = GetGateFromJson<RYGateImpl<Prec, Space>>::get(j);
        else if (type == "RZ") gate = GetGateFromJson<RZGateImpl<Prec, Space>>::get(j);
        else if (type == "U1") gate = GetGateFromJson<U1GateImpl<Prec, Space>>::get(j);
        else if (type == "U2") gate = GetGateFromJson<U2GateImpl<Prec, Space>>::get(j);
        else if (type == "U3") gate = GetGateFromJson<U3GateImpl<Prec, Space>>::get(j);
        else if (type == "Swap") gate = GetGateFromJson<SwapGateImpl<Prec, Space>>::get(j);
        else if (type == "Pauli") gate = GetGateFromJson<PauliGateImpl<Prec, Space>>::get(j);
        else if (type == "PauliRotation") gate = GetGateFromJson<PauliRotationGateImpl<Prec, Space>>::get(j);
        else if (type == "Probabilistic") gate = GetGateFromJson<ProbabilisticGateImpl<Prec, Space>>::get(j);
        // clang-format on
    }
};

}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using Gate = internal::GatePtr<internal::GateBase<Prec, Space>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
#define DEF_GATE_BASE(GATE_TYPE, PREC, SPACE, DESCRIPTION)                                         \
    nb::class_<GATE_TYPE<PREC, SPACE>>(m, #GATE_TYPE, DESCRIPTION)                                 \
        .def(nb::init<Gate<PREC, SPACE>>(), "Downcast from Gate.")                                 \
        .def("gate_type", &GATE_TYPE<PREC, SPACE>::gate_type, "Get gate type as `GateType` enum.") \
        .def(                                                                                      \
            "target_qubit_list",                                                                   \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->target_qubit_list(); },          \
            "Get target qubits as `list[int]`. **Control qubits are not included.**")              \
        .def(                                                                                      \
            "control_qubit_list",                                                                  \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->control_qubit_list(); },         \
            "Get control qubits as `list[int]`.")                                                  \
        .def(                                                                                      \
            "operand_qubit_list",                                                                  \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->operand_qubit_list(); },         \
            "Get target and control qubits as `list[int]`.")                                       \
        .def(                                                                                      \
            "target_qubit_mask",                                                                   \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->target_qubit_mask(); },          \
            "Get target qubits as mask. **Control qubits are not included.**")                     \
        .def(                                                                                      \
            "control_qubit_mask",                                                                  \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->control_qubit_mask(); },         \
            "Get control qubits as mask.")                                                         \
        .def(                                                                                      \
            "operand_qubit_mask",                                                                  \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->operand_qubit_mask(); },         \
            "Get target and control qubits as mask.")                                              \
        .def(                                                                                      \
            "get_inverse",                                                                         \
            [](const GATE_TYPE<PREC, SPACE>& gate) {                                               \
                auto inv = gate->get_inverse();                                                    \
                if (!inv) nb::none();                                                              \
                return Gate<PREC, SPACE>(inv);                                                     \
            },                                                                                     \
            "Generate inverse gate as `Gate` type. If not exists, return None.")                   \
        .def(                                                                                      \
            "update_quantum_state",                                                                \
            [](const GATE_TYPE<PREC, SPACE>& gate, StateVector<PREC, SPACE>& state_vector) {       \
                gate->update_quantum_state(state_vector);                                          \
            },                                                                                     \
            "state_vector"_a,                                                                      \
            "Apply gate to `state_vector`. `state_vector` in args is directly updated.")           \
        .def(                                                                                      \
            "update_quantum_state",                                                                \
            [](const GATE_TYPE<PREC, SPACE>& gate, StateVectorBatched<PREC, SPACE>& states) {      \
                gate->update_quantum_state(states);                                                \
            },                                                                                     \
            "states"_a,                                                                            \
            "Apply gate to `states`. `states` in args is directly updated.")                       \
        .def(                                                                                      \
            "get_matrix",                                                                          \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->get_matrix(); },                 \
            "Get matrix representation of the gate.")                                              \
        .def(                                                                                      \
            "to_string",                                                                           \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->to_string(""); },                \
            "Get string representation of the gate.")                                              \
        .def(                                                                                      \
            "__str__",                                                                             \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return gate->to_string(""); },                \
            "Get string representation of the gate.")                                              \
        .def(                                                                                      \
            "to_json",                                                                             \
            [](const GATE_TYPE<PREC, SPACE>& gate) { return Json(gate).dump(); },                  \
            "Get JSON representation of the gate.")                                                \
        .def(                                                                                      \
            "load_json",                                                                           \
            [](GATE_TYPE<PREC, SPACE>& gate, const std::string& str) {                             \
                gate = nlohmann::json::parse(str);                                                 \
            },                                                                                     \
            "json_str"_a,                                                                          \
            "Read an object from the JSON representation of the gate.")
#define DEF_GATE(GATE_TYPE, PRECISION, SPACE, DESCRIPTION, GATE_BASE_DEF)                        \
    GATE_BASE_DEF.def(nb::init<GATE_TYPE<PRECISION, SPACE>>(), "Upcast from `" #GATE_TYPE "`."); \
    DEF_GATE_BASE(                                                                               \
        GATE_TYPE,                                                                               \
        PRECISION,                                                                               \
        SPACE,                                                                                   \
        DESCRIPTION                                                                              \
        "\n\nNotes\n\tUpcast is required to use gate-general functions (ex: add to Circuit).")   \
        .def(nb::init<Gate<PRECISION, SPACE>>())

void bind_gate_gate_hpp_without_precision_and_space(nb::module_& m) {
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
        .value("DenseMatrix", GateType::DenseMatrix)
        .value("Probabilistic", GateType::Probabilistic);
}

template <Precision Prec, ExecutionSpace Space>
nb::class_<Gate<Prec, Space>> bind_gate_gate_hpp(nb::module_& m) {
    return DEF_GATE_BASE(Gate,
                         Prec,
                         Space,
                         "General class of QuantumGate.\n\nNotes\n\tDowncast to required to use "
                         "gate-specific functions.")
        .def(nb::init<Gate<Prec, Space>>(), "Just copy shallowly.");
}
}  // namespace internal
#endif

}  // namespace scaluq

// Include all gate header files for the correct definition of concept GateImpl
#include "./gate_matrix.hpp"
#include "./gate_pauli.hpp"
#include "./gate_probabilistic.hpp"
#include "./gate_standard.hpp"
