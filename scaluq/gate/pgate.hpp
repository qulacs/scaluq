#pragma once

#include "../state/state_vector.hpp"
#include "../types.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
// forward declarations
class PGateBase;

template <typename T>
concept PGateImpl = std::derived_from<T, PGateBase>;

class PRXGateImpl;
class PRYGateImpl;
class PRZGateImpl;
class PPauliRotationGateImpl;

template <PGateImpl T>
class PGatePtr;
}  // namespace internal
using PGate = internal::PGatePtr<internal::PGateBase>;

enum class PGateType { Unknown, PRX, PRY, PRZ, PPauliRotation };

template <internal::PGateImpl T>
constexpr PGateType get_pgate_type() {
    if constexpr (std::is_same_v<T, internal::PGateBase>) return PGateType::Unknown;
    if constexpr (std::is_same_v<T, internal::PRXGateImpl>) return PGateType::PRX;
    if constexpr (std::is_same_v<T, internal::PRYGateImpl>) return PGateType::PRY;
    if constexpr (std::is_same_v<T, internal::PRZGateImpl>) return PGateType::PRZ;
    if constexpr (std::is_same_v<T, internal::PPauliRotationGateImpl>)
        return PGateType::PPauliRotation;
    static_assert("unknown PGateImpl");
    return PGateType::Unknown;
}

namespace internal {
class PGateBase {
protected:
    double _pcoef;

public:
    virtual ~PGateBase() = default;

    PGateBase(double pcoef = 1.) : _pcoef(pcoef) {}

    [[nodiscard]] double pcoef() { return _pcoef; }

    [[nodiscard]] virtual std::vector<UINT> get_target_qubit_list() const = 0;
    [[nodiscard]] virtual std::vector<UINT> get_control_qubit_list() const = 0;

    [[nodiscard]] virtual PGate copy() const = 0;
    [[nodiscard]] virtual PGate get_inverse() const = 0;
    [[nodiscard]] virtual std::optional<ComplexMatrix> get_matrix(double param) const = 0;

    virtual void update_quantum_state(StateVector& state_vector, double param) const = 0;
};

template <PGateImpl T>
class PGatePtr {
    friend class PGateFactory;
    template <PGateImpl U>
    friend class PGatePtr;

private:
    std::shared_ptr<T> _pgate_ptr;
    PGateType _pgate_type;

public:
    PGatePtr() : _pgate_ptr(nullptr), _pgate_type(get_pgate_type<T>()) {}
    PGatePtr(const PGatePtr& pgate) = default;
    template <PGateImpl U>
    PGatePtr(const std::shared_ptr<U>& pgate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _pgate_type = get_pgate_type<T>();
            _pgate_ptr = pgate_ptr;
        } else if constexpr (std::is_same_v<T, internal::PGateBase>) {
            // upcast
            _pgate_type = get_pgate_type<U>();
            _pgate_ptr = std::static_pointer_cast<T>(pgate_ptr);
        } else {
            // downcast
            _pgate_type = get_pgate_type<T>();
            if (!(_pgate_ptr = std::dynamic_pointer_cast<T>(pgate_ptr))) {
                throw std::runtime_error("invalid gate cast");
            }
        }
    }
    template <PGateImpl U>
    PGatePtr(const PGatePtr<U>& pgate) {
        if constexpr (std::is_same_v<T, U>) {
            _pgate_type = pgate._pgate_type;
            _pgate_ptr = pgate._pgate_ptr;
        } else if constexpr (std::is_same_v<T, internal::PGateBase>) {
            // upcast
            _pgate_type = pgate._pgate_type;
            _pgate_ptr = std::static_pointer_cast<T>(pgate._pgate_ptr);
        } else {
            // downcast
            if (pgate._pgate_type != get_pgate_type<T>()) {
                throw std::runtime_error("invalid gate cast");
            }
            _pgate_type = pgate._pgate_type;
            _pgate_ptr = std::static_pointer_cast<T>(pgate._pgate_ptr);
        }
    }

    PGateType pgate_type() const { return _pgate_type; }

    T* operator->() const {
        if (!_pgate_ptr) {
            throw std::runtime_error("PGatePtr::operator->(): PGate is Null");
        }
        return _pgate_ptr.get();
    }
};
}  // namespace internal

}  // namespace scaluq
