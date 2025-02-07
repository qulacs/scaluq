#pragma once

#include "param_gate_pauli.hpp"
#include "param_gate_probablistic.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
class ParamGateFactory {
public:
    template <ParamGateImpl T, typename... Args>
    static ParamGate<T::Prec> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {
template <Precision Prec>
inline ParamGate<Prec> ParamRX(std::uint64_t target,
                               double param_coef = 1.,
                               const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRXGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec>
inline ParamGate<Prec> ParamRY(std::uint64_t target,
                               double param_coef = 1.,
                               const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRYGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec>
inline ParamGate<Prec> ParamRZ(std::uint64_t target,
                               double param_coef = 1.,
                               const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRZGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec>
inline ParamGate<Prec> ParamPauliRotation(const PauliOperator<Prec>& pauli,
                                          double param_coef = 1.,
                                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamPauliRotationGateImpl<Prec>>(
        internal::vector_to_mask(controls), pauli, static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec>
inline ParamGate<Prec> ParamProbablistic(
    const std::vector<double>& distribution,
    const std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>>& gate_list) {
    return internal::ParamGateFactory::create_gate<internal::ParamProbablisticGateImpl<Prec>>(
        distribution, gate_list);
}
}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_param_gate_factory(nb::module_& mgate) {
    mgate.def("ParamRX",
              &gate::ParamRX<Prec>,
              "Generate general ParamGate class instance of ParamRX.",
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamRY",
              &gate::ParamRY<Prec>,
              "Generate general ParamGate class instance of ParamRY.",
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamRZ",
              &gate::ParamRZ<Prec>,
              "Generate general ParamGate class instance of ParamRZ.",
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamPauliRotation",
              &gate::ParamPauliRotation<Prec>,
              "Generate general ParamGate class instance of ParamPauliRotation.",
              "pauli"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("ParamProbablistic",
              &gate::ParamProbablistic<Prec>,
              "Generate general ParamGate class instance of ParamProbablistic.");
    mgate.def(
        "ParamProbablistic",
        [](const std::vector<std::pair<double, std::variant<Gate<Prec>, ParamGate<Prec>>>>&
               prob_gate_list) {
            std::vector<double> distribution;
            std::vector<std::variant<Gate<Prec>, ParamGate<Prec>>> gate_list;
            distribution.reserve(prob_gate_list.size());
            gate_list.reserve(prob_gate_list.size());
            for (const auto& [prob, gate] : prob_gate_list) {
                distribution.push_back(prob);
                gate_list.push_back(gate);
            }
            return gate::ParamProbablistic<Prec>(distribution, gate_list);
        },
        "Generate general ParamGate class instance of ParamProbablistic.");
}
}  // namespace internal
#endif
}  // namespace scaluq
