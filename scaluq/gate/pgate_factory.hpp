#pragma once

#include "pgate_one_qubit.hpp"
#include "pgate_pauli.hpp"

namespace scaluq {
namespace internal {
class PGateFactory {
public:
    template <PGateImpl T, typename... Args>
    static PGate create_gate(Args... args) {
        return {std::make_shared<T>(args...)};
    }
};
}  // namespace internal

inline PGate PRX(UINT target) {
    return internal::PGateFactory::create_gate<internal::PRXGateImpl>(target);
}
inline PGate PRY(UINT target) {
    return internal::PGateFactory::create_gate<internal::PRXGateImpl>(target);
}
inline PGate PRZ(UINT target) {
    return internal::PGateFactory::create_gate<internal::PRXGateImpl>(target);
}
inline PGate PPauliRotation(const PauliOperator& pauli) {
    return internal::PGateFactory::create_gate<internal::PPauliRotationGateImpl>(pauli);
}
}  // namespace scaluq
