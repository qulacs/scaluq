#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
// forward declarations
class ParamGateBase;

template <typename T>
concept ParamGateImpl = std::derived_from<T, ParamGateBase>;

class ParamRXGateImpl;
class ParamRYGateImpl;
class ParamRZGateImpl;
class ParamPauliRotationGateImpl;
class ParamProbablisticGateImpl;

template <ParamGateImpl T>
class ParamGatePtr;
}  // namespace internal
using ParamGate = internal::ParamGatePtr<internal::ParamGateBase>;

enum class ParamGateType {
    Unknown,
    ParamRX,
    ParamRY,
    ParamRZ,
    ParamPauliRotation,
    ParamProbablistic,
    Error
};

template <internal::ParamGateImpl T>
constexpr ParamGateType get_param_gate_type() {
    using TWithoutConst = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<TWithoutConst, internal::ParamGateBase>)
        return ParamGateType::Unknown;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRXGateImpl>)
        return ParamGateType::ParamRX;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRYGateImpl>)
        return ParamGateType::ParamRY;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRZGateImpl>)
        return ParamGateType::ParamRZ;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamPauliRotationGateImpl>)
        return ParamGateType::ParamPauliRotation;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamProbablisticGateImpl>)
        return ParamGateType::ParamProbablistic;
    else
        static_assert(internal::lazy_false_v<T>, "unknown GateImpl");
}

namespace internal {
class ParamGateBase {
protected:
    std::uint64_t _target_mask, _control_mask;
    double _pcoef;
    void check_qubit_mask_within_bounds(const StateVector& state_vector) const {
        std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
        if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: ParamGate::update_quantum_state(StateVector& state): "
                "Target/Control qubit exceeds the number of qubits in the system.");
        }
    }

    void check_qubit_mask_within_bounds(const StateVectorBatched& states) const {
        std::uint64_t full_mask = (1ULL << states.n_qubits()) - 1;
        if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: ParamGate::update_quantum_state(StateVector& state): "
                "Target/Control qubit exceeds the number of qubits in the system.");
        }
    }

    std::string get_qubit_info_as_string(const std::string& indent) const {
        std::ostringstream ss;
        auto targets = target_qubit_list();
        auto controls = control_qubit_list();
        ss << indent << "  Parameter Coefficient: " << _pcoef << "\n";
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
    ParamGateBase(std::uint64_t target_mask, std::uint64_t control_mask, double param_coef = 1.)
        : _target_mask(target_mask), _control_mask(control_mask), _pcoef(param_coef) {
        if (_target_mask & _control_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: ParamGate::ParamGate(std::uint64_t target_mask, std::uint64_t "
                "control_mask) : Target and control qubits must not overlap.");
        }
    }
    virtual ~ParamGateBase() = default;

    [[nodiscard]] double param_coef() const { return _pcoef; }

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

    [[nodiscard]] virtual ParamGate get_inverse() const = 0;
    [[nodiscard]] virtual internal::ComplexMatrix get_matrix(double param) const = 0;

    virtual void update_quantum_state(StateVector& state_vector, double param) const = 0;
    virtual void update_quantum_state(StateVectorBatched& states,
                                      std::vector<double> params) const = 0;

    [[nodiscard]] virtual std::string to_string(const std::string& indent = "") const = 0;
    [[nodiscard]] virtual bool is_noise() { return false; }
};

template <ParamGateImpl T>
class ParamGatePtr {
    friend class ParamGateFactory;
    template <ParamGateImpl U>
    friend class ParamGatePtr;

private:
    std::shared_ptr<const T> _param_gate_ptr;
    ParamGateType _param_gate_type;

public:
    ParamGatePtr() : _param_gate_ptr(nullptr), _param_gate_type(get_param_gate_type<T>()) {}
    ParamGatePtr(const ParamGatePtr& param_gate) = default;
    template <ParamGateImpl U>
    ParamGatePtr(const std::shared_ptr<const U>& param_gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _param_gate_type = get_param_gate_type<T>();
            _param_gate_ptr = param_gate_ptr;
        } else if constexpr (std::is_same_v<T, internal::ParamGateBase>) {
            // upcast
            _param_gate_type = get_param_gate_type<U>();
            _param_gate_ptr = std::static_pointer_cast<const T>(param_gate_ptr);
        } else {
            // downcast
            _param_gate_type = get_param_gate_type<T>();
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
        } else if constexpr (std::is_same_v<T, internal::ParamGateBase>) {
            // upcast
            _param_gate_type = param_gate._param_gate_type;
            _param_gate_ptr = std::static_pointer_cast<const T>(param_gate._param_gate_ptr);
        } else {
            // downcast
            if (param_gate._param_gate_type != get_param_gate_type<T>()) {
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
};
}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
#define DEF_PARAM_GATE_BASE(PARAM_GATE_TYPE, DESCRIPTION)                                         \
    nb::class_<PARAM_GATE_TYPE>(m, #PARAM_GATE_TYPE, DESCRIPTION)                                 \
        .def("param_gate_type",                                                                   \
             &PARAM_GATE_TYPE::param_gate_type,                                                   \
             "Get parametric gate type as `ParamGateType` enum.")                                 \
        .def(                                                                                     \
            "param_coef",                                                                         \
            [](const PARAM_GATE_TYPE& gate) { return gate->param_coef(); },                       \
            "Get coefficient of parameter.")                                                      \
        .def(                                                                                     \
            "target_qubit_list",                                                                  \
            [](const PARAM_GATE_TYPE& gate) { return gate->target_qubit_list(); },                \
            "Get target qubits as `list[int]`. **Control qubits is not included.**")              \
        .def(                                                                                     \
            "control_qubit_list",                                                                 \
            [](const PARAM_GATE_TYPE& gate) { return gate->control_qubit_list(); },               \
            "Get control qubits as `list[int]`.")                                                 \
        .def(                                                                                     \
            "operand_qubit_list",                                                                 \
            [](const PARAM_GATE_TYPE& gate) { return gate->operand_qubit_list(); },               \
            "Get target and control qubits as `list[int]`.")                                      \
        .def(                                                                                     \
            "target_qubit_mask",                                                                  \
            [](const PARAM_GATE_TYPE& gate) { return gate->target_qubit_mask(); },                \
            "Get target qubits as mask. **Control qubits is not included.**")                     \
        .def(                                                                                     \
            "control_qubit_mask",                                                                 \
            [](const PARAM_GATE_TYPE& gate) { return gate->control_qubit_mask(); },               \
            "Get control qubits as mask.")                                                        \
        .def(                                                                                     \
            "operand_qubit_mask",                                                                 \
            [](const PARAM_GATE_TYPE& gate) { return gate->operand_qubit_mask(); },               \
            "Get target and control qubits as mask.")                                             \
        .def(                                                                                     \
            "get_inverse",                                                                        \
            [](const PARAM_GATE_TYPE& param_gate) { return param_gate->get_inverse(); },          \
            "Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.")  \
        .def(                                                                                     \
            "update_quantum_state",                                                               \
            [](const PARAM_GATE_TYPE& param_gate, StateVector& state_vector, double param) {      \
                param_gate->update_quantum_state(state_vector, param);                            \
            },                                                                                    \
            "Apply gate to `state_vector` with holding the parameter. `state_vector` in args is " \
            "directly updated.")                                                                  \
        .def(                                                                                     \
            "update_quantum_state",                                                               \
            [](const PARAM_GATE_TYPE& param_gate,                                                 \
               StateVectorBatched& states,                                                        \
               std::vector<double> params) { param_gate->update_quantum_state(states, params); }, \
            "Apply gate to `states` with holding the parameter. `states` in args is directly "    \
            "updated.")                                                                           \
        .def(                                                                                     \
            "get_matrix",                                                                         \
            [](const PARAM_GATE_TYPE& gate, double param) { return gate->get_matrix(param); },    \
            "Get matrix representation of the gate with holding the parameter.")

nb::class_<ParamGate> param_gate_base_def;

#define DEF_PARAM_GATE(PARAM_GATE_TYPE, DESCRIPTION)                                            \
    ::scaluq::internal::param_gate_base_def.def(nb::init<PARAM_GATE_TYPE>(),                    \
                                                "Upcast from `" #PARAM_GATE_TYPE "`.");         \
    DEF_PARAM_GATE_BASE(                                                                        \
        PARAM_GATE_TYPE,                                                                        \
        DESCRIPTION                                                                             \
        "\n\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).") \
        .def(nb::init<ParamGate>())

void bind_gate_param_gate_hpp(nb::module_& m) {
    nb::enum_<ParamGateType>(m, "ParamGateType", "Enum of ParamGate Type.")
        .value("ParamRX", ParamGateType::ParamRX)
        .value("ParamRY", ParamGateType::ParamRY)
        .value("ParamRZ", ParamGateType::ParamRZ)
        .value("ParamPauliRotation", ParamGateType::ParamPauliRotation);

    param_gate_base_def = DEF_PARAM_GATE_BASE(
        ParamGate,
        "General class of parametric quantum gate.\n\n.. note:: Downcast to requred to use "
        "gate-specific functions.");
}

}  // namespace internal
#endif
}  // namespace scaluq
