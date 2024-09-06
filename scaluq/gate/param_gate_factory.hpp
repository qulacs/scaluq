#pragma once

#include "param_gate_pauli.hpp"
#include "param_gate_probablistic.hpp"
#include "param_gate_standard.hpp"

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
inline ParamGate ParamRX(std::uint64_t target,
                         double param_coef = 1.,
                         const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRXGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
inline ParamGate ParamRY(std::uint64_t target,
                         double param_coef = 1.,
                         const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRYGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
inline ParamGate ParamRZ(std::uint64_t target,
                         double param_coef = 1.,
                         const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
// まだ
inline ParamGate ParamPauliRotation(const PauliOperator& pauli,
                                    double param_coef = 1.,
                                    const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamPauliRotationGateImpl>(
        internal::vector_to_mask(controls), pauli, param_coef);
}
inline ParamGate ParamProbablistic(const std::vector<double>& distribution,
                                   const std::vector<std::variant<Gate, ParamGate>>& gate_list) {
    return internal::ParamGateFactory::create_gate<internal::ParamProbablisticGateImpl>(
        distribution, gate_list);
}
}  // namespace gate
}  // namespace scaluq
