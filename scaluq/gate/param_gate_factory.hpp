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
        return {std::make_shared<T>(args...)};
    }
};
inline std::pair<std::vector<double>, std::vector<std::variant<Gate, ParamGate>>>
decompose_dist_pgate() {
    return {};
}
template <typename... Tail>
inline std::pair<std::vector<double>, std::vector<std::variant<Gate, ParamGate>>>
decompose_dist_gate(const std::pair<double, Gate> head, Tail&&... tail) {
    auto ret = decompose_dist_gate(tail...);
    ret.first.push_back(head.first);
    ret.second.push_back(head.second);
    return ret;
}
template <typename... Tail>
inline std::pair<std::vector<double>, std::vector<std::variant<Gate, ParamGate>>>
decompose_dist_gate(const std::pair<double, ParamGate> head, Tail&&... tail) {
    auto ret = decompose_dist_gate(tail...);
    ret.first.push_back(head.first);
    ret.second.push_back(head.second);
    return ret;
}
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
inline ParamGate PProbablistic(const std::vector<double>& distribution,
                               const std::vector<std::variant<Gate, ParamGate>>& gate_list) {
    return internal::ParamGateFactory::create_gate<internal::PProbablisticGateImpl>(distribution,
                                                                                    gate_list);
}
inline ParamGate PProbablistic(const std::pair<double, Gate>& head) {
    return internal::ParamGateFactory::create_gate<internal::PProbablisticGateImpl>(
        std::vector{head.first}, std::vector{std::variant<Gate, ParamGate>{head.second}});
}
inline ParamGate PProbablistic(const std::pair<double, ParamGate>& head) {
    return internal::ParamGateFactory::create_gate<internal::PProbablisticGateImpl>(
        std::vector{head.first}, std::vector{std::variant<Gate, ParamGate>{head.second}});
}
template <class... Args>
inline ParamGate PProbablistic(const std::pair<double, Gate>& head, Args&&... args) {
    auto [distribution, gate_list] = internal::decompose_dist_gate(head, args...);
    std::ranges::reverse(distribution);
    std::ranges::reverse(gate_list);
    return internal::ParamGateFactory::create_gate<internal::PProbablisticGateImpl>(distribution,
                                                                                    gate_list);
}
template <class... Args>
inline ParamGate PProbablistic(const std::pair<double, ParamGate>& head, Args&&... args) {
    auto [distribution, gate_list] = internal::decompose_dist_gate(head, args...);
    std::ranges::reverse(distribution);
    std::ranges::reverse(gate_list);
    return internal::ParamGateFactory::create_gate<internal::PProbablisticGateImpl>(distribution,
                                                                                    gate_list);
}
}  // namespace gate
}  // namespace scaluq
