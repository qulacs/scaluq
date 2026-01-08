#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"

namespace scaluq {
namespace internal {
// forward declarations

template <Precision Prec>
class ParamGateBase;

template <Precision Prec>
class ParamRXGateImpl;
template <Precision Prec>
class ParamRYGateImpl;
template <Precision Prec>
class ParamRZGateImpl;
template <Precision Prec>
class ParamPauliRotationGateImpl;
template <Precision Prec>
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

template <typename T, Precision Prec>
constexpr ParamGateType get_param_gate_type() {
    using TWithoutConst = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<TWithoutConst, internal::ParamGateBase<Prec>>)
        return ParamGateType::Unknown;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRXGateImpl<Prec>>)
        return ParamGateType::ParamRX;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRYGateImpl<Prec>>)
        return ParamGateType::ParamRY;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamRZGateImpl<Prec>>)
        return ParamGateType::ParamRZ;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamPauliRotationGateImpl<Prec>>)
        return ParamGateType::ParamPauliRotation;
    else if constexpr (std::is_same_v<TWithoutConst, internal::ParamProbabilisticGateImpl<Prec>>)
        return ParamGateType::ParamProbabilistic;
    else
        static_assert(internal::lazy_false_v<T>, "unknown GateImpl");
}

namespace internal {
template <Precision _Prec>
class ParamGateBase : public std::enable_shared_from_this<ParamGateBase<_Prec>> {
public:
    constexpr static Precision Prec = _Prec;
    using FloatType = Float<Prec>;
    using ComplexType = Complex<Prec>;

protected:
    std::uint64_t _target_mask, _control_mask, _control_value_mask;
    FloatType _pcoef;
    void check_qubit_mask_within_bounds(
        const StateVector<Prec, ExecutionSpace::Host>& state_vector) const;
    void check_qubit_mask_within_bounds(
        const StateVectorBatched<Prec, ExecutionSpace::Host>& states) const;
#ifdef SCALUQ_USE_CUDA
    void check_qubit_mask_within_bounds(
        const StateVector<Prec, ExecutionSpace::Default>& state_vector) const;
    void check_qubit_mask_within_bounds(
        const StateVectorBatched<Prec, ExecutionSpace::Default>& states) const;
#endif  // SCALUQ_USE_CUDA

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

    [[nodiscard]] virtual std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const = 0;
    [[nodiscard]] virtual ComplexMatrix get_matrix(double param) const = 0;

    virtual void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                                      double param) const = 0;
    virtual void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                                      std::vector<double> params) const = 0;
#ifdef SCALUQ_USE_CUDA
    virtual void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                                      double param) const = 0;
    virtual void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                                      std::vector<double> params) const = 0;
#endif  // SCALUQ_USE_CUDA

    [[nodiscard]] virtual std::string to_string(const std::string& indent = "") const = 0;

    virtual void get_as_json(Json& j) const { j = Json{{"type", "Unknown"}}; }
};

template <typename T>
concept ParamGateImpl = std::derived_from<T, ParamGateBase<T::Prec>>;

template <ParamGateImpl T>
struct GetParamGateFromJson {
    static std::shared_ptr<const T> get(const Json&) {
        throw std::runtime_error("GetParamGateFromJson<T>::get() is not implemented");
    }
};
#define DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(Impl)           \
    template <Precision Prec>                                        \
    struct GetParamGateFromJson<Impl<Prec>> {                        \
        static std::shared_ptr<const Impl<Prec>> get(const Json& j); \
    };
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamRXGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamRYGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamRZGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamPauliRotationGateImpl)
DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION(ParamProbabilisticGateImpl)
#undef DECLARE_GET_FROM_JSON_PARTIAL_SPECIALIZATION

template <ParamGateImpl T>
class ParamGatePtr {
    friend class ParamGateFactory;
    template <ParamGateImpl U>
    friend class ParamGatePtr;
    constexpr static Precision Prec = T::Prec;
    using FloatType = Float<Prec>;
    using ComplexType = Complex<Prec>;

private:
    std::shared_ptr<const T> _param_gate_ptr;
    ParamGateType _param_gate_type;

public:
    ParamGatePtr() : _param_gate_ptr(nullptr), _param_gate_type(get_param_gate_type<T, Prec>()) {}
    template <ParamGateImpl U>
    ParamGatePtr(const std::shared_ptr<const U>& param_gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _param_gate_type = get_param_gate_type<T, Prec>();
            _param_gate_ptr = param_gate_ptr;
        } else if constexpr (std::is_same_v<T, ParamGateBase<Prec>>) {
            // upcast
            _param_gate_type = get_param_gate_type<U, Prec>();
            _param_gate_ptr = std::static_pointer_cast<const T>(param_gate_ptr);
        } else {
            // downcast
            _param_gate_type = get_param_gate_type<T, Prec>();
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
        } else if constexpr (std::is_same_v<T, ParamGateBase<Prec>>) {
            // upcast
            _param_gate_type = param_gate._param_gate_type;
            _param_gate_ptr = std::static_pointer_cast<const T>(param_gate._param_gate_ptr);
        } else {
            // downcast
            if (param_gate._param_gate_type != get_param_gate_type<T, Prec>()) {
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
        if (type == "ParamRX") gate = GetParamGateFromJson<ParamRXGateImpl<Prec>>::get(j);
        else if (type == "ParamRY") gate = GetParamGateFromJson<ParamRYGateImpl<Prec>>::get(j);
        else if (type == "ParamRZ") gate = GetParamGateFromJson<ParamRZGateImpl<Prec>>::get(j);
        else if (type == "ParamPauliRotation") gate = GetParamGateFromJson<ParamPauliRotationGateImpl<Prec>>::get(j);
        else if (type == "ParamProbabilistic") gate = GetParamGateFromJson<ParamProbabilisticGateImpl<Prec>>::get(j);
        // clang-format on
    }
};
}  // namespace internal

template <Precision Prec>
using ParamGate = internal::ParamGatePtr<internal::ParamGateBase<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <typename GateT, Precision Prec>
void register_param_gate_common_methods(nb::class_<GateT>& c) {
    c.def(nb::init<GateT>(), "Downcast from ParamGate.")
        .def("param_gate_type",
             &GateT::param_gate_type,
             "Get parametric gate type as `ParamGateType` enum.")
        .def(
            "param_coef",
            [](const GateT& gate) { return gate->param_coef(); },
            "Get coefficient of parameter.")
        .def(
            "target_qubit_list",
            [](const GateT& gate) { return gate->target_qubit_list(); },
            "Get target qubits as `list[int]`. **Control qubits is not included.**")
        .def(
            "control_qubit_list",
            [](const GateT& gate) { return gate->control_qubit_list(); },
            "Get control qubits as `list[int]`.")
        .def(
            "operand_qubit_list",
            [](const GateT& gate) { return gate->operand_qubit_list(); },
            "Get target and control qubits as `list[int]`.")
        .def(
            "target_qubit_mask",
            [](const GateT& gate) { return gate->target_qubit_mask(); },
            "Get target qubits as mask. **Control qubits is not included.**")
        .def(
            "control_qubit_mask",
            [](const GateT& gate) { return gate->control_qubit_mask(); },
            "Get control qubits as mask.")
        .def(
            "operand_qubit_mask",
            [](const GateT& gate) { return gate->operand_qubit_mask(); },
            "Get target and control qubits as mask.")
        .def(
            "get_inverse",
            [](const GateT& param_gate) -> nb::object {
                auto inv = param_gate->get_inverse();
                if (!inv) return nb::none();
                return nb::cast(ParamGate<Prec>(inv));
            },
            "Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.")
        .def(
            "update_quantum_state",
            [](const GateT& gate,
               StateVector<Prec, ExecutionSpace::Host>& state_vector,
               double param) { gate->update_quantum_state(state_vector, param); },
            "state"_a,
            "param"_a,
            "Apply gate to `state_vector` with holding the parameter. `state_vector` in args is "
            "directly updated.")
        .def(
            "update_quantum_state",
            [](const GateT& gate,
               StateVectorBatched<Prec, ExecutionSpace::Host>& states,
               std::vector<double> params) { gate->update_quantum_state(states, params); },
            "states"_a,
            "params"_a,
            "Apply gate to `states` with holding the parameters. `states` in args is directly "
            "updated.")
#ifdef SCALUQ_USE_CUDA
        .def(
            "update_quantum_state",
            [](const GateT& gate,
               StateVector<Prec, ExecutionSpace::Default>& state_vector,
               double param) { gate->update_quantum_state(state_vector, param); },
            "state"_a,
            "param"_a,
            "Apply gate to `state_vector` with holding the parameter. `state_vector` in args is "
            "directly updated.")
        .def(
            "update_quantum_state",
            [](const GateT& gate,
               StateVectorBatched<Prec, ExecutionSpace::Default>& states,
               std::vector<double> params) { gate->update_quantum_state(states, params); },
            "states"_a,
            "params"_a,
            "Apply gate to `states` with holding the parameters. `states` in args is directly "
            "updated.")
#endif  // SCALUQ_USE_CUDA
        .def(
            "get_matrix",
            [](const GateT& gate, double param) { return gate->get_matrix(param); },
            "param"_a,
            "Get matrix representation of the gate with holding the parameter.")
        .def(
            "to_string",
            [](const GateT& gate) { return gate->to_string(""); },
            "Get string representation of the gate.")
        .def(
            "__str__",
            [](const GateT& gate) { return gate->to_string(""); },
            "Get string representation of the gate.")
        .def(
            "to_json",
            [](const GateT& gate) { return Json(gate).dump(); },
            "Get JSON representation of the gate.")
        .def(
            "load_json",
            [](GateT& gate, const std::string& str) { gate = nlohmann::json::parse(str); },
            "json_str"_a,
            "Read an object from the JSON representation of the gate.");
}

void bind_gate_param_gate_hpp_without_precision_and_space(nb::module_& m) {
    nb::enum_<ParamGateType>(m, "ParamGateType", "Enum of ParamGate Type.")
        .value("ParamRX", ParamGateType::ParamRX)
        .value("ParamRY", ParamGateType::ParamRY)
        .value("ParamRZ", ParamGateType::ParamRZ)
        .value("ParamPauliRotation", ParamGateType::ParamPauliRotation);
}

template <Precision Prec>
nb::class_<ParamGate<Prec>> bind_gate_param_gate_hpp(nb::module_& m) {
    using GateT = ParamGate<Prec>;
    const char* description =
        "General class of parametric quantum gate.\n\nNotes:\n\t"
        "Downcast to required to use gate-specific functions.";
    auto c = nb::class_<GateT>(m, "ParamGate", description);
    register_param_gate_common_methods<GateT, Prec>(c);
    c.def(nb::init<GateT>(), "Just copy shallowly.");
    return c;
}

template <class SpecificGateType, Precision Prec>
nb::class_<SpecificGateType> bind_specific_param_gate(nb::module_& m,
                                                      nb::class_<ParamGate<Prec>>& base_class,
                                                      const char* name,
                                                      const char* description) {
    using BaseGateT = ParamGate<Prec>;
    base_class.def(nb::init<SpecificGateType>(),
                   "param_gate"_a,
                   ("Upcast from `" + std::string(name) + "`.").c_str());
    std::string full_description = std::string(description) +
                                   "\n\nNotes:\n\tUpcast is required to use gate-general functions "
                                   "(ex: add to Circuit).";
    auto c = nb::class_<SpecificGateType>(m, name, full_description.c_str());
    register_param_gate_common_methods<SpecificGateType, Prec>(c);
    c.def(nb::init<BaseGateT>());
    return c;
}

}  // namespace internal
#endif
}  // namespace scaluq

// Include all gate header files for the correct definition of concept GateImpl
#include "./param_gate_pauli.hpp"
#include "./param_gate_probabilistic.hpp"
#include "./param_gate_standard.hpp"
