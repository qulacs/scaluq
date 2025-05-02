#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"

namespace scaluq {
namespace internal {
// forward declarations

template <Precision Prec, ExecutionSpace Space>
class ParamGateBase;

template <Precision Prec, ExecutionSpace Space>
class ParamRXGateImpl;
template <Precision Prec, ExecutionSpace Space>
class ParamRYGateImpl;
template <Precision Prec, ExecutionSpace Space>
class ParamRZGateImpl;
template <Precision Prec, ExecutionSpace Space>
class ParamPauliRotationGateImpl;
template <Precision Prec, ExecutionSpace Space>
class ParamProbabilisticGateImpl;

}  // namespace internal

enum class ParamGateType {
    Unknown,
    ParamRX,
    ParamRY,
    ParamRZ,
    ParamPauliRotation,
    ParamProbabilistic,
    Error
};

template <typename T, Precision Prec, ExecutionSpace Space>
constexpr ParamGateType get_param_gate_type() {
    using TWithoutConst = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<TWithoutConst, internal::ParamGateBase<Prec, Space>>)
        return ParamGateType::Unknown;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRXGateImpl<Prec, Space>>)
        return ParamGateType::ParamRX;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRYGateImpl<Prec, Space>>)
        return ParamGateType::ParamRY;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRZGateImpl<Prec, Space>>)
        return ParamGateType::ParamRZ;
    else if constexpr (std::is_same_v<TWithoutConst,
                                      internal::ParamPauliRotationGateImpl<Prec, Space>>)
        return ParamGateType::ParamPauliRotation;
    else if constexpr (std::is_same_v<TWithoutConst,
                                      internal::ParamProbabilisticGateImpl<Prec, Space>>)
        return ParamGateType::ParamProbabilistic;
    else
        static_assert(internal::lazy_false_v<T>, "unknown GateImpl");
}

namespace internal {
template <Precision _Prec, ExecutionSpace _Space>
class ParamGateBase : public std::enable_shared_from_this<ParamGateBase<_Prec, _Space>> {
public:
    constexpr static Precision Prec = _Prec;
    constexpr static ExecutionSpace Space = _Space;
    using FloatType = Float<Prec>;
    using ComplexType = Complex<Prec>;

protected:
    std::uint64_t _target_mask, _control_mask, _control_value_mask;
    FloatType _pcoef;
    void check_qubit_mask_within_bounds(const StateVector<Prec, Space>& state_vector) const;
    void check_qubit_mask_within_bounds(const StateVectorBatched<Prec, Space>& states) const;

    std::string get_qubit_info_as_string(const std::string& indent) const;

public:
    ParamGateBase(std::uint64_t target_mask,
                  std::uint64_t control_mask,
                  std::uint64_t control_value_mask,
                  Float<Prec> param_coef = 1);
    virtual ~ParamGateBase() = default;

    [[nodiscard]] double param_coef() const { return _pcoef; }

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

    [[nodiscard]] virtual std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const = 0;
    [[nodiscard]] virtual ComplexMatrix get_matrix(double param) const = 0;

    virtual void update_quantum_state(StateVector<Prec, Space>& state_vector,
                                      double param) const = 0;
    virtual void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                                      std::vector<double> params) const = 0;

    [[nodiscard]] virtual std::string to_string(const std::string& indent = "") const = 0;

    virtual void get_as_json(Json& j) const { j = Json{{"type", "Unknown"}}; }
};

template <typename T>
concept ParamGateImpl = std::derived_from<T, ParamGateBase<T::Prec, T::Space>>;

template <ParamGateImpl T>
struct GetParamGateFromJson {
    static std::shared_ptr<const T> get(const Json&) {
        throw std::runtime_error("GetParamGateFromJson<T>::get() is not implemented");
    }
};
#define DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(Impl)                  \
    template <Precision Prec, ExecutionSpace Space>                         \
    struct GetParamGateFromJson<Impl<Prec, Space>> {                        \
        static std::shared_ptr<const Impl<Prec, Space>> get(const Json& j); \
    };
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamRXGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamRYGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamRZGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamPauliRotationGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamProbabilisticGateImpl)

template <ParamGateImpl T>
class ParamGatePtr {
    friend class ParamGateFactory;
    template <ParamGateImpl U>
    friend class ParamGatePtr;
    constexpr static Precision Prec = T::Prec;
    constexpr static ExecutionSpace Space = T::Space;
    using FloatType = Float<Prec>;
    using ComplexType = Complex<Prec>;

private:
    std::shared_ptr<const T> _param_gate_ptr;
    ParamGateType _param_gate_type;

public:
    ParamGatePtr()
        : _param_gate_ptr(nullptr), _param_gate_type(get_param_gate_type<T, Prec, Space>()) {}
    template <ParamGateImpl U>
    ParamGatePtr(const std::shared_ptr<const U>& param_gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _param_gate_type = get_param_gate_type<T, Prec, Space>();
            _param_gate_ptr = param_gate_ptr;
        } else if constexpr (std::is_same_v<T, ParamGateBase<Prec, Space>>) {
            // upcast
            _param_gate_type = get_param_gate_type<U, Prec, Space>();
            _param_gate_ptr = std::static_pointer_cast<const T>(param_gate_ptr);
        } else {
            // downcast
            _param_gate_type = get_param_gate_type<T, Prec, Space>();
            if (!(_param_gate_ptr = std::dynamic_pointer_cast<const T>(param_gate_ptr))) {
                throw std::runtime_error("invalid gate cast");
            }
        }
    }
    template <ParamGateImpl U>
    ParamGatePtr(const ParamGatePtr<U>& param_gate) {
        if constexpr (std::is_same_v<T, U>) {
            _param_gate_type = param_gate._param_gate_type;
            _param_gate_ptr = param_gate._param_gate_ptr;
        } else if constexpr (std::is_same_v<T, ParamGateBase<Prec, Space>>) {
            // upcast
            _param_gate_type = param_gate._param_gate_type;
            _param_gate_ptr = std::static_pointer_cast<const T>(param_gate._param_gate_ptr);
        } else {
            // downcast
            if (param_gate._param_gate_type != get_param_gate_type<T, Prec, Space>()) {
                throw std::runtime_error("invalid gate cast");
            }
            _param_gate_type = param_gate._param_gate_type;
            _param_gate_ptr = std::static_pointer_cast<const T>(param_gate._param_gate_ptr);
        }
    }

    ParamGateType param_gate_type() const { return _param_gate_type; }

    const T* operator->() const {
        if (!_param_gate_ptr) {
            throw std::runtime_error("ParamGatePtr::operator->(): ParamGate is Null");
        }
        return _param_gate_ptr.get();
    }

    friend std::ostream& operator<<(std::ostream& os, ParamGatePtr gate) {
        os << gate->to_string();
        return os;
    }

    friend void to_json(Json& j, const ParamGatePtr& gate) { gate->get_as_json(j); }

    friend void from_json(const Json& j, ParamGatePtr& gate) {
        std::string type = j.at("type");

        // clang-format off
        if (type == "ParamRX") gate = GetParamGateFromJson<ParamRXGateImpl<Prec, Space>>::get(j);
        else if (type == "ParamRY") gate = GetParamGateFromJson<ParamRYGateImpl<Prec, Space>>::get(j);
        else if (type == "ParamRZ") gate = GetParamGateFromJson<ParamRZGateImpl<Prec, Space>>::get(j);
        else if (type == "ParamPauliRotation") gate = GetParamGateFromJson<ParamPauliRotationGateImpl<Prec, Space>>::get(j);
        else if (type == "ParamProbabilistic") gate = GetParamGateFromJson<ParamProbabilisticGateImpl<Prec, Space>>::get(j);
        // clang-format on
    }
};
}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using ParamGate = internal::ParamGatePtr<internal::ParamGateBase<Prec, Space>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
#define DEF_PARAM_GATE_BASE(PARAM_GATE_TYPE, PRECISION, SPACE, DESCRIPTION)                       \
    nb::class_<PARAM_GATE_TYPE<PRECISION, SPACE>>(m, #PARAM_GATE_TYPE, DESCRIPTION)               \
        .def(nb::init<PARAM_GATE_TYPE<PRECISION, SPACE>>(), "Downcast from ParamGate.")           \
        .def("param_gate_type",                                                                   \
             &PARAM_GATE_TYPE<PRECISION, SPACE>::param_gate_type,                                 \
             "Get parametric gate type as `ParamGateType` enum.")                                 \
        .def(                                                                                     \
            "param_coef",                                                                         \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) { return gate->param_coef(); },     \
            "Get coefficient of parameter.")                                                      \
        .def(                                                                                     \
            "target_qubit_list",                                                                  \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) {                                   \
                return gate->target_qubit_list();                                                 \
            },                                                                                    \
            "Get target qubits as `list[int]`. **Control qubits is not included.**")              \
        .def(                                                                                     \
            "control_qubit_list",                                                                 \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) {                                   \
                return gate->control_qubit_list();                                                \
            },                                                                                    \
            "Get control qubits as `list[int]`.")                                                 \
        .def(                                                                                     \
            "operand_qubit_list",                                                                 \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) {                                   \
                return gate->operand_qubit_list();                                                \
            },                                                                                    \
            "Get target and control qubits as `list[int]`.")                                      \
        .def(                                                                                     \
            "target_qubit_mask",                                                                  \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) {                                   \
                return gate->target_qubit_mask();                                                 \
            },                                                                                    \
            "Get target qubits as mask. **Control qubits is not included.**")                     \
        .def(                                                                                     \
            "control_qubit_mask",                                                                 \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) {                                   \
                return gate->control_qubit_mask();                                                \
            },                                                                                    \
            "Get control qubits as mask.")                                                        \
        .def(                                                                                     \
            "operand_qubit_mask",                                                                 \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) {                                   \
                return gate->operand_qubit_mask();                                                \
            },                                                                                    \
            "Get target and control qubits as mask.")                                             \
        .def(                                                                                     \
            "get_inverse",                                                                        \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& param_gate) {                             \
                auto inv = param_gate->get_inverse();                                             \
                if (!inv) nb::none();                                                             \
                return ParamGate<PRECISION, SPACE>(inv);                                          \
            },                                                                                    \
            "Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.")  \
        .def(                                                                                     \
            "update_quantum_state",                                                               \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& param_gate,                               \
               StateVector<PRECISION, SPACE>& state_vector,                                       \
               double param) { param_gate->update_quantum_state(state_vector, param); },          \
            "state"_a,                                                                            \
            "param"_a,                                                                            \
            "Apply gate to `state_vector` with holding the parameter. `state_vector` in args is " \
            "directly updated.")                                                                  \
        .def(                                                                                     \
            "update_quantum_state",                                                               \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& param_gate,                               \
               StateVectorBatched<PRECISION, SPACE>& states,                                      \
               std::vector<double> params) { param_gate->update_quantum_state(states, params); }, \
            "states"_a,                                                                           \
            "params"_a,                                                                           \
            "Apply gate to `states` with holding the parameter. `states` in args is directly "    \
            "updated.")                                                                           \
        .def(                                                                                     \
            "get_matrix",                                                                         \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate, double param) {                     \
                return gate->get_matrix(param);                                                   \
            },                                                                                    \
            "param"_a,                                                                            \
            "Get matrix representation of the gate with holding the parameter.")                  \
        .def(                                                                                     \
            "to_string",                                                                          \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) { return gate->to_string(""); },    \
            "Get string representation of the gate.")                                             \
        .def(                                                                                     \
            "__str__",                                                                            \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) { return gate->to_string(""); },    \
            "Get string representation of the gate.")                                             \
        .def(                                                                                     \
            "to_json",                                                                            \
            [](const PARAM_GATE_TYPE<PRECISION, SPACE>& gate) { return Json(gate).dump(); },      \
            "Get JSON representation of the gate.")                                               \
        .def(                                                                                     \
            "load_json",                                                                          \
            [](PARAM_GATE_TYPE<PRECISION, SPACE>& gate, const std::string& str) {                 \
                gate = nlohmann::json::parse(str);                                                \
            },                                                                                    \
            "json_str"_a,                                                                         \
            "Read an object from the JSON representation of the gate.")
#define DEF_PARAM_GATE(PARAM_GATE_TYPE, PRECISION, SPACE, DESCRIPTION, PARAM_GATE_BASE_DEF) \
    PARAM_GATE_BASE_DEF.def(nb::init<PARAM_GATE_TYPE<PRECISION, SPACE>>(),                  \
                            "param_gate"_a,                                                 \
                            "Upcast from `" #PARAM_GATE_TYPE "`.");                         \
    DEF_PARAM_GATE_BASE(PARAM_GATE_TYPE,                                                    \
                        PRECISION,                                                          \
                        SPACE,                                                              \
                        DESCRIPTION                                                         \
                        "\n\nNotes:\n\tUpcast is required to use gate-general functions "   \
                        "(ex: add to Circuit).")                                            \
        .def(nb::init<ParamGate<PRECISION, SPACE>>())

void bind_gate_param_gate_hpp_without_precision_and_space(nb::module_& m) {
    nb::enum_<ParamGateType>(m, "ParamGateType", "Enum of ParamGate Type.")
        .value("ParamRX", ParamGateType::ParamRX)
        .value("ParamRY", ParamGateType::ParamRY)
        .value("ParamRZ", ParamGateType::ParamRZ)
        .value("ParamPauliRotation", ParamGateType::ParamPauliRotation);
}

template <Precision Prec, ExecutionSpace Space>
nb::class_<ParamGate<Prec, Space>> bind_gate_param_gate_hpp(nb::module_& m) {
    return DEF_PARAM_GATE_BASE(ParamGate,
                               Prec,
                               Space,
                               "General class of parametric quantum gate.\n\nNotes:\n\t"
                               "Downcast to required to use "
                               "gate-specific functions.")
        .def(nb::init<ParamGate<Prec, Space>>(), "Just copy shallowly.");
}

}  // namespace internal
#endif
}  // namespace scaluq
