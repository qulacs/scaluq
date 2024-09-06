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

namespace internal {

template <ParamGateImpl T>
std::string gate_to_string(const ParamGatePtr<T>& obj, std::uint32_t depth = 0) {
    std::ostringstream ss;
    std::string indent(depth * 2, ' ');

    if (obj.param_gate_type() == ParamGateType::ParamProbablistic) {
        const auto prob_gate = ParamProbablisticGate(obj);
        const auto distribution = prob_gate->distribution();
        const auto gates = prob_gate->gate_list();
        ss << indent << "Gate Type: Probablistic\n";
        for (std::size_t i = 0; i < distribution.size(); ++i) {
            ss << indent << "  --------------------\n";
            ss << indent << "  Probability: " << distribution[i] << "\n";
            std::visit(
                [&](auto&& arg) {
                    ss << gate_to_string(arg, depth + 1)
                       << (i == distribution.size() - 1 ? "" : "\n");
                },
                gates[i]);
        }
        return ss.str();
    }

    auto targets = internal::mask_to_vector(obj->target_qubit_mask());
    auto controls = internal::mask_to_vector(obj->control_qubit_mask());
    auto param_coef = obj->param_coef();

    ss << indent << "Gate Type: ";
    switch (obj.param_gate_type()) {
        case ParamGateType::ParamRX:
            ss << "ParamRX";
            break;
        case ParamGateType::ParamRY:
            ss << "ParamRY";
            break;
        case ParamGateType::ParamRZ:
            ss << "ParamRZ";
            break;
        case ParamGateType::ParamPauliRotation:
            ss << "ParamPauliRotation";
            break;
        case ParamGateType::Unknown:
        default:
            ss << "Undefined";
            break;
    }

    ss << "\n";
    ss << indent << "  Parameter Coefficient: " << param_coef << "\n";
    ss << indent << "  Target Qubits: {";
    for (std::uint32_t i = 0; i < targets.size(); ++i)
        ss << targets[i] << (i == targets.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}";
    return ss.str();
}
}  // namespace internal

template <internal::ParamGateImpl T>
std::ostream& operator<<(std::ostream& os, const internal::ParamGatePtr<T>& obj) {
    os << internal::gate_to_string(obj);
    return os;
}

}  // namespace scaluq
