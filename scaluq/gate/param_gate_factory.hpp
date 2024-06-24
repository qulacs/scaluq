#pragma once

#include "param_gate_one_qubit.hpp"
#include "param_gate_pauli.hpp"

namespace scaluq {
namespace internal {
class ParamGateFactory {
public:
    template <ParamGateImpl T, typename... Args>
    static ParamGate create_gate(Args... args) {
        return {std::make_shared<T>(args...)};
    }
};
}  // namespace internal
namespace gate {
inline ParamGate PRX(UINT target, double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PRXGateImpl>(target, pcoef);
}
inline ParamGate PRY(UINT target, double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PRYGateImpl>(target, pcoef);
}
inline ParamGate PRZ(UINT target, double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PRZGateImpl>(target, pcoef);
}
inline ParamGate PPauliRotation(const PauliOperator& pauli, double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PPauliRotationGateImpl>(pauli, pcoef);
}
}  // namespace gate
}  // namespace scaluq
