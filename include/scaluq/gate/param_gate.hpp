#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"
#include "gate.hpp"

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
    void check_qubit_mask_within_bounds(
        const StateVector<Prec, ExecutionSpace::HostSerial>& state_vector) const;
    void check_qubit_mask_within_bounds(
        const StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states) const;
#ifdef SCALUQ_USE_DEVICE
    void check_qubit_mask_within_bounds(
        const StateVector<Prec, ExecutionSpace::Default>& state_vector) const;
    void check_qubit_mask_within_bounds(
        const StateVectorBatched<Prec, ExecutionSpace::Default>& states) const;
#endif  // SCALUQ_USE_DEVICE

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

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                              double param) const {
        ClassicalRegister classical_register(0);
        std::mt19937_64 random_engine(std::random_device{}());
        ExecutionContext<Prec, ExecutionSpace::Host> context{
            state_vector, classical_register, random_engine};
        update_quantum_state(context, param);
    }
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                              ClassicalRegister& classical_register,
                              double param,
                              std::optional<std::uint64_t> seed = std::nullopt) const {
        std::mt19937_64 random_engine(resolve_seed(seed));
        ExecutionContext<Prec, ExecutionSpace::Host> context{
            state_vector, classical_register, random_engine};
        update_quantum_state(context, param);
    }
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                              const std::vector<double>& params) const {
        ClassicalRegisterBatched classical_register(0, states.batch_size());
        std::mt19937_64 random_engine(std::random_device{}());
        ExecutionContextBatched<Prec, ExecutionSpace::Host> context{
            states, classical_register, random_engine};
        update_quantum_state(context, params);
    }
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                              ClassicalRegisterBatched& classical_register,
                              const std::vector<double>& params,
                              std::optional<std::uint64_t> seed = std::nullopt) const {
        if (classical_register.batch_size() != states.batch_size()) {
            throw std::runtime_error(
                "ParamGateBase::update_quantum_state(StateVectorBatched&, "
                "ClassicalRegisterBatched&, ...): batch size mismatch.");
        }
        if (params.size() != states.batch_size()) {
            throw std::runtime_error(
                "ParamGateBase::update_quantum_state(StateVectorBatched&, "
                "ClassicalRegisterBatched&, ...): parameter size mismatch.");
        }
        std::mt19937_64 random_engine(resolve_seed(seed));
        ExecutionContextBatched<Prec, ExecutionSpace::Host> context{
            states, classical_register, random_engine};
        update_quantum_state(context, params);
    }
    void update_quantum_state(StateVector<Prec, ExecutionSpace::HostSerial>& state_vector,
                              double param) const {
        ClassicalRegister classical_register(0);
        std::mt19937_64 random_engine(std::random_device{}());
        ExecutionContext<Prec, ExecutionSpace::HostSerial> context{
            state_vector, classical_register, random_engine};
        update_quantum_state(context, param);
    }
    void update_quantum_state(StateVector<Prec, ExecutionSpace::HostSerial>& state_vector,
                              ClassicalRegister& classical_register,
                              double param,
                              std::optional<std::uint64_t> seed = std::nullopt) const {
        std::mt19937_64 random_engine(resolve_seed(seed));
        ExecutionContext<Prec, ExecutionSpace::HostSerial> context{
            state_vector, classical_register, random_engine};
        update_quantum_state(context, param);
    }
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
                              const std::vector<double>& params) const {
        ClassicalRegisterBatched classical_register(0, states.batch_size());
        std::mt19937_64 random_engine(std::random_device{}());
        ExecutionContextBatched<Prec, ExecutionSpace::HostSerial> context{
            states, classical_register, random_engine};
        update_quantum_state(context, params);
    }
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
                              ClassicalRegisterBatched& classical_register,
                              const std::vector<double>& params,
                              std::optional<std::uint64_t> seed = std::nullopt) const {
        if (classical_register.batch_size() != states.batch_size()) {
            throw std::runtime_error(
                "ParamGateBase::update_quantum_state(StateVectorBatched&, "
                "ClassicalRegisterBatched&, ...): batch size mismatch.");
        }
        if (params.size() != states.batch_size()) {
            throw std::runtime_error(
                "ParamGateBase::update_quantum_state(StateVectorBatched&, "
                "ClassicalRegisterBatched&, ...): parameter size mismatch.");
        }
        std::mt19937_64 random_engine(resolve_seed(seed));
        ExecutionContextBatched<Prec, ExecutionSpace::HostSerial> context{
            states, classical_register, random_engine};
        update_quantum_state(context, params);
    }
#ifdef SCALUQ_USE_DEVICE
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                              double param) const {
        ClassicalRegister classical_register(0);
        std::mt19937_64 random_engine(std::random_device{}());
        ExecutionContext<Prec, ExecutionSpace::Default> context{
            state_vector, classical_register, random_engine};
        update_quantum_state(context, param);
    }
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                              ClassicalRegister& classical_register,
                              double param,
                              std::optional<std::uint64_t> seed = std::nullopt) const {
        std::mt19937_64 random_engine(resolve_seed(seed));
        ExecutionContext<Prec, ExecutionSpace::Default> context{
            state_vector, classical_register, random_engine};
        update_quantum_state(context, param);
    }
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                              const std::vector<double>& params) const {
        ClassicalRegisterBatched classical_register(0, states.batch_size());
        std::mt19937_64 random_engine(std::random_device{}());
        ExecutionContextBatched<Prec, ExecutionSpace::Default> context{
            states, classical_register, random_engine};
        update_quantum_state(context, params);
    }
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                              ClassicalRegisterBatched& classical_register,
                              const std::vector<double>& params,
                              std::optional<std::uint64_t> seed = std::nullopt) const {
        if (classical_register.batch_size() != states.batch_size()) {
            throw std::runtime_error(
                "ParamGateBase::update_quantum_state(StateVectorBatched&, "
                "ClassicalRegisterBatched&, ...): batch size mismatch.");
        }
        if (params.size() != states.batch_size()) {
            throw std::runtime_error(
                "ParamGateBase::update_quantum_state(StateVectorBatched&, "
                "ClassicalRegisterBatched&, ...): parameter size mismatch.");
        }
        std::mt19937_64 random_engine(resolve_seed(seed));
        ExecutionContextBatched<Prec, ExecutionSpace::Default> context{
            states, classical_register, random_engine};
        update_quantum_state(context, params);
    }
#endif  // SCALUQ_USE_DEVICE

    virtual void update_quantum_state(ExecutionContext<Prec, ExecutionSpace::Host>& context,
                                      double param) const = 0;
    virtual void update_quantum_state(ExecutionContextBatched<Prec, ExecutionSpace::Host>& context,
                                      const std::vector<double>& params) const = 0;
    virtual void update_quantum_state(ExecutionContext<Prec, ExecutionSpace::HostSerial>& context,
                                      double param) const = 0;
    virtual void update_quantum_state(
        ExecutionContextBatched<Prec, ExecutionSpace::HostSerial>& context,
        const std::vector<double>& params) const = 0;
#ifdef SCALUQ_USE_DEVICE
    virtual void update_quantum_state(ExecutionContext<Prec, ExecutionSpace::Default>& context,
                                      double param) const = 0;
    virtual void update_quantum_state(
        ExecutionContextBatched<Prec, ExecutionSpace::Default>& context,
        const std::vector<double>& params) const = 0;
#endif

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
using ParamValueVariant = std::variant<std::monostate, double, std::vector<double>>;

template <typename GateT, Precision Prec, ExecutionSpace Space>
void update_param_gate_state(const GateT& gate,
                             StateVector<Prec, Space>* state,
                             const ParamValueVariant& param,
                             ClassicalRegisterVariant classical_register,
                             std::optional<std::uint64_t> seed) {
    const auto* single_param = std::get_if<double>(&param);
    if (!single_param) {
        throw std::runtime_error(
            "ParamGate::update_quantum_state(): StateVector requires a float parameter.");
    }
    std::visit(
        Overloaded{[&](std::monostate) {
                       if (seed) {
                           ClassicalRegister reg(0);
                           gate->update_quantum_state(*state, reg, *single_param, *seed);
                       } else {
                           gate->update_quantum_state(*state, *single_param);
                       }
                   },
                   [&](ClassicalRegister* reg) {
                       gate->update_quantum_state(
                           *state, *reg, *single_param, seed.value_or(std::random_device{}()));
                   },
                   [&](ClassicalRegisterBatched*) {
                       throw std::runtime_error(
                           "ParamGate::update_quantum_state(): ClassicalRegisterBatched cannot "
                           "be used with StateVector.");
                   }},
        classical_register);
}

template <typename GateT, Precision Prec, ExecutionSpace Space>
void update_param_gate_state(const GateT& gate,
                             StateVectorBatched<Prec, Space>* states,
                             const ParamValueVariant& param,
                             ClassicalRegisterVariant classical_register,
                             std::optional<std::uint64_t> seed) {
    const auto* batched_params = std::get_if<std::vector<double>>(&param);
    if (!batched_params) {
        throw std::runtime_error(
            "ParamGate::update_quantum_state(): StateVectorBatched requires a sequence of float "
            "parameters.");
    }
    std::visit(
        Overloaded{[&](std::monostate) {
                       if (seed) {
                           ClassicalRegisterBatched reg(0, states->batch_size());
                           gate->update_quantum_state(*states, reg, *batched_params, *seed);
                       } else {
                           gate->update_quantum_state(*states, *batched_params);
                       }
                   },
                   [&](ClassicalRegister*) {
                       throw std::runtime_error(
                           "ParamGate::update_quantum_state(): ClassicalRegister cannot be used "
                           "with StateVectorBatched.");
                   },
                   [&](ClassicalRegisterBatched* reg) {
                       gate->update_quantum_state(
                           *states, *reg, *batched_params, seed.value_or(std::random_device{}()));
                   }},
        classical_register);
}

template <typename GateT, Precision Prec>
void register_param_gate_common_methods(nb::class_<GateT>& c) {
    constexpr bool is_base_gate = std::is_same_v<GateT, ParamGate<Prec>>;

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
            "Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.");
    constexpr const char* update_signature =
        "def update_quantum_state(self, state: StateVector | StateVectorBatched, "
        "*, param: float | Sequence[float], classical_register: ClassicalRegister | "
        "ClassicalRegisterBatched | None = None, seed: int | None = None) -> None";
    auto update_doc_str = ([]() {
        auto ds =
            DocString()
                .desc("Apply gate to `state`. `state` in args is directly updated.")
                .desc("Optionally pass a matching classical register and seed.")
                .arg("state",
                     "StateVector | StateVectorBatched",
                     "State vector to be updated. Can be `StateVector` or `StateVectorBatched`.")
                .arg("param",
                     "float | Sequence[float]",
                     "Parameter value(s) for the gate. Should be a single float for `StateVector` "
                     "and a sequence of floats (one for each batch) for `StateVectorBatched`.")
                .arg("classical_register",
                     "ClassicalRegister | ClassicalRegisterBatched | None",
                     "Classical register to be used in the gate. If not provided, a new register "
                     "with appropriate size is created and used internally.")
                .arg("seed",
                     "int | None",
                     "Seed for random number generator. If not provided, a random seed is "
                     "generated using `std::random_device`.");
        if constexpr (is_base_gate) {
            ds.ex(DocString::Code({">>> import math",
                                   ">>> state = StateVector(2)",
                                   ">>> state.set_computational_basis(0)",
                                   ">>> ParamRX(0).update_quantum_state(state, param=math.pi / 4)",
                                   ">>> print(state)",
                                   "Qubit Count : 2",
                                   "Dimension : 4",
                                   "State vector : ",
                                   "  00 : (0.92388,0)",
                                   "  01 : (0,-0.382683)",
                                   "  10 : (0,0)",
                                   "  11 : (0,0)"}));
            ds.ex(DocString::Code(
                {">>> import math",
                 ">>> states = StateVectorBatched(2, 1)",
                 ">>> states.set_computational_basis(0)",
                 ">>> ParamRX(0).update_quantum_state(states, param=[math.pi / 4, -math.pi / 4])",
                 ">>> print(states)",
                 "Qubit Count : 1",
                 "Dimension : 2",
                 "--------------------",
                 "Batch_id : 0",
                 "State vector : ",
                 "  0 : (0.92388,0)",
                 "  1 : (0,-0.382683)",
                 "--------------------",
                 "Batch_id : 1",
                 "State vector : ",
                 "  0 : (0.92388,0)",
                 "  1 : (0,0.382683)",
                 "<BLANKLINE>"}));
        }
        return ds.build_as_google_style();
    }());
    c.def(
        "update_quantum_state",
        [](const GateT& gate,
           GateStateVariant<Prec> state,
           ParamValueVariant param,
           ClassicalRegisterVariant classical_register,
           std::optional<std::uint64_t> seed) {
            std::visit(
                [&](auto* state_ptr) {
                    update_param_gate_state<GateT, Prec>(
                        gate, state_ptr, param, classical_register, seed);
                },
                state);
        },
        "state"_a,
        nb::kw_only(),
        "param"_a,
        "classical_register"_a = std::monostate{},
        "seed"_a = std::nullopt,
        nb::sig(update_signature),
        update_doc_str.c_str());
    c.def(
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
