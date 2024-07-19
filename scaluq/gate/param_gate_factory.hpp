#pragma once

#include "param_gate_one_qubit.hpp"
#include "param_gate_pauli.hpp"
#include "param_gate_probablistic.hpp"

namespace scaluq {
namespace internal {
class ParamGateFactory {
public:
    template <ParamGateImpl T, typename... Args>
    static ParamGate create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {
inline ParamGate PRX(UINT target, const std::vector<UINT>& controls, double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PRXGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), pcoef);
}
inline ParamGate PRY(UINT target, const std::vector<UINT>& controls, double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PRYGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), pcoef);
}
inline ParamGate PRZ(UINT target, const std::vector<UINT>& controls, double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PRZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), pcoef);
}
// まだ
inline ParamGate PPauliRotation(const std::vector<UINT>& controls,
                                const PauliOperator& pauli,
                                double pcoef = 1.) {
    return internal::ParamGateFactory::create_gate<internal::PPauliRotationGateImpl>(
        internal::vector_to_mask(controls), pauli, pcoef);
}
inline ParamGate PProbablistic(const std::vector<double>& distribution,
                               const std::vector<std::variant<Gate, ParamGate>>& gate_list) {
    return internal::ParamGateFactory::create_gate<internal::PProbablisticGateImpl>(distribution,
                                                                                    gate_list);
}
}  // namespace gate
}  // namespace scaluq
