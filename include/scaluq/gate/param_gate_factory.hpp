#pragma once

#include "param_gate_pauli.hpp"
#include "param_gate_probablistic.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
class ParamGateFactory {
public:
    template <ParamGateImpl T, typename... Args>
    static ParamGate<typename T::Fp, typename T::Sp> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {
template <std::floating_point Fp, ExecutionSpace Sp>
inline ParamGate<Fp, Sp> ParamRX(std::uint64_t target,
                                 Fp param_coef = 1.,
                                 const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRXGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline ParamGate<Fp, Sp> ParamRY(std::uint64_t target,
                                 Fp param_coef = 1.,
                                 const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRYGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline ParamGate<Fp, Sp> ParamRZ(std::uint64_t target,
                                 Fp param_coef = 1.,
                                 const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRZGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline ParamGate<Fp, Sp> ParamPauliRotation(const PauliOperator<Fp, Sp>& pauli,
                                            Fp param_coef = 1.,
                                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamPauliRotationGateImpl<Fp, Sp>>(
        internal::vector_to_mask(controls), pauli, param_coef);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline ParamGate<Fp, Sp> ParamProbablistic(
    const std::vector<Fp>& distribution,
    const std::vector<std::variant<Gate<Fp, Sp>, ParamGate<Fp, Sp>>>& gate_list) {
    return internal::ParamGateFactory::create_gate<internal::ParamProbablisticGateImpl<Fp, Sp>>(
        distribution, gate_list);
}
}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_gate_param_gate_factory(nb::module_& mgate) {
    mgate.def("ParamRX",
              &gate::ParamRX<Fp>,
              "Generate general ParamGate class instance of ParamRX.",
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamRY",
              &gate::ParamRY<Fp>,
              "Generate general ParamGate class instance of ParamRY.",
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamRZ",
              &gate::ParamRZ<Fp>,
              "Generate general ParamGate class instance of ParamRZ.",
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamPauliRotation",
              &gate::ParamPauliRotation<Fp>,
              "Generate general ParamGate class instance of ParamPauliRotation.",
              "pauli"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamProbablistic",
              &gate::ParamProbablistic<Fp>,
              "Generate general ParamGate class instance of ParamProbablistic.");
    mgate.def(
        "ParamProbablistic",
        [](const std::vector<std::pair<Fp, std::variant<Gate<Fp>, ParamGate<Fp>>>>&
               prob_gate_list) {
            std::vector<Fp> distribution;
            std::vector<std::variant<Gate<Fp>, ParamGate<Fp>>> gate_list;
            distribution.reserve(prob_gate_list.size());
            gate_list.reserve(prob_gate_list.size());
            for (const auto& [prob, gate] : prob_gate_list) {
                distribution.push_back(prob);
                gate_list.push_back(gate);
            }
            return gate::ParamProbablistic<Fp>(distribution, gate_list);
        },
        "Generate general ParamGate class instance of ParamProbablistic.");
}
}  // namespace internal
#endif
}  // namespace scaluq
