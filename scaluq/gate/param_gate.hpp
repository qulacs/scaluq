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

class PRXGateImpl;
class PRYGateImpl;
class PRZGateImpl;
class PPauliRotationGateImpl;

template <ParamGateImpl T>
class ParamGatePtr;
}  // namespace internal
using ParamGate = internal::ParamGatePtr<internal::ParamGateBase>;

enum class ParamGateType { Unknown, PRX, PRY, PRZ, PPauliRotation };

template <internal::ParamGateImpl T>
constexpr ParamGateType get_param_gate_type() {
    if constexpr (std::is_same_v<T, internal::ParamGateBase>) return ParamGateType::Unknown;
    if constexpr (std::is_same_v<T, internal::PRXGateImpl>) return ParamGateType::PRX;
    if constexpr (std::is_same_v<T, internal::PRYGateImpl>) return ParamGateType::PRY;
    if constexpr (std::is_same_v<T, internal::PRZGateImpl>) return ParamGateType::PRZ;
    if constexpr (std::is_same_v<T, internal::PPauliRotationGateImpl>)
        return ParamGateType::PPauliRotation;
    static_assert("unknown ParamGateImpl");
    return ParamGateType::Unknown;
}

namespace internal {
class ParamGateBase {
protected:
    double _pcoef;

public:
    virtual ~ParamGateBase() = default;

    ParamGateBase(double pcoef = 1.) : _pcoef(pcoef) {}

    [[nodiscard]] double pcoef() { return _pcoef; }

    [[nodiscard]] virtual std::vector<UINT> get_target_qubit_list() const = 0;
    [[nodiscard]] virtual std::vector<UINT> get_control_qubit_list() const = 0;

    [[nodiscard]] virtual ParamGate copy() const = 0;
    [[nodiscard]] virtual ParamGate get_inverse() const = 0;
    [[nodiscard]] virtual std::optional<ComplexMatrix> get_matrix(double param) const = 0;

    virtual void update_quantum_state(StateVector& state_vector, double param) const = 0;
};

template <ParamGateImpl T>
class ParamGatePtr {
    friend class ParamGateFactory;
    template <ParamGateImpl U>
    friend class ParamGatePtr;

private:
    std::shared_ptr<T> _param_gate_ptr;
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
            _param_gate_ptr = std::static_pointer_cast<T>(param_gate_ptr);
        } else {
            // downcast
            _param_gate_type = get_param_gate_type<T>();
            if (!(_param_gate_ptr = std::dynamic_pointer_cast<T>(param_gate_ptr))) {
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
            _param_gate_ptr = std::static_pointer_cast<T>(param_gate._param_gate_ptr);
        } else {
            // downcast
            if (param_gate._param_gate_type != get_param_gate_type<T>()) {
                throw std::runtime_error("invalid gate cast");
            }
            _param_gate_type = param_gate._param_gate_type;
            _param_gate_ptr = std::static_pointer_cast<T>(param_gate._param_gate_ptr);
        }
    }

    ParamGateType param_gate_type() const { return _param_gate_type; }

    T* operator->() const {
        if (!_param_gate_ptr) {
            throw std::runtime_error("ParamGatePtr::operator->(): ParamGate is Null");
        }
        return _param_gate_ptr.get();
    }
};
}  // namespace internal

}  // namespace scaluq
