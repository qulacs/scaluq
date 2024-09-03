#pragma once

#include "../state/state_vector.hpp"
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

template <ParamGateImpl T>
class ParamGatePtr;
}  // namespace internal
using ParamGate = internal::ParamGatePtr<internal::ParamGateBase>;

enum class ParamGateType { Unknown, ParamRX, ParamRY, ParamRZ, ParamPauliRotation };

template <internal::ParamGateImpl T>
constexpr ParamGateType get_param_gate_type() {
    if constexpr (std::is_same_v<T, internal::ParamGateBase>) return ParamGateType::Unknown;
    if constexpr (std::is_same_v<T, internal::ParamRXGateImpl>) return ParamGateType::ParamRX;
    if constexpr (std::is_same_v<T, internal::ParamRYGateImpl>) return ParamGateType::ParamRY;
    if constexpr (std::is_same_v<T, internal::ParamRZGateImpl>) return ParamGateType::ParamRZ;
    if constexpr (std::is_same_v<T, internal::ParamPauliRotationGateImpl>)
        return ParamGateType::ParamPauliRotation;
    static_assert("unknown ParamGateImpl");
    return ParamGateType::Unknown;
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

public:
    ParamGateBase(std::uint64_t target_mask, std::uint64_t control_mask, double param_coef = 1.)
        : _target_mask(target_mask), _control_mask(control_mask), _pcoef(param_coef) {
        if (_target_mask & _control_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: ParamGate::ParamGate(std::uint64_t target_mask, std::uint64_t "
                "control_mask) : Target and "
                "control qubits must not overlap.");
        }
    }
    virtual ~ParamGateBase() = default;

    [[nodiscard]] double param_coef() { return _pcoef; }

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
    [[nodiscard]] virtual ComplexMatrix get_matrix(double param) const = 0;

    virtual void update_quantum_state(StateVector& state_vector, double param) const = 0;
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
    ParamGatePtr(const std::shared_ptr<U>& param_gate_ptr) {
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

    friend std::ostream& operator<<(std::ostream& os, const ParamGatePtr& obj) {
        if (!obj._param_gate_ptr) {
            os << "Gate Type: Null";
            return os;
        }
        auto targets = internal::mask_to_vector(obj->target_qubit_mask());
        auto controls = internal::mask_to_vector(obj->control_qubit_mask());
        os << "Gate Type: ";
        switch (obj.param_gate_type()) {
            case ParamGateType::ParamRX:
                os << "ParamRX";
                break;
            case ParamGateType::ParamRY:
                os << "ParamRY";
                break;
            case ParamGateType::ParamRZ:
                os << "ParamRZ";
                break;
            case ParamGateType::ParamPauliRotation:
                os << "ParamPauliRotation";
                break;
            default:
                os << "Undefined";
                break;
        }
        os << "\n"
              "Target Qubits: {";
        for (std::uint32_t i = 0; i < targets.size(); ++i)
            os << targets[i] << (i == targets.size() - 1 ? "" : ", ");
        os << "}\n"
              "Control Qubits: {";
        for (std::uint32_t i = 0; i < controls.size(); ++i)
            os << controls[i] << (i == controls.size() - 1 ? "" : ", ");
        os << "}";
        return os;
    }
};
}  // namespace internal

}  // namespace scaluq
