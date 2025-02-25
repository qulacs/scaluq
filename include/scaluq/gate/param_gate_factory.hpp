#pragma once

#include "param_gate_pauli.hpp"
#include "param_gate_probablistic.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
class ParamGateFactory {
public:
    template <ParamGateImpl T, typename... Args>
    static ParamGate<typename T::Fp> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {
template <std::floating_point Fp>
inline ParamGate<Fp> ParamRX(std::uint64_t target,
                             Fp param_coef = 1.,
                             const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRXGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
template <std::floating_point Fp>
inline ParamGate<Fp> ParamRY(std::uint64_t target,
                             Fp param_coef = 1.,
                             const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRYGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
template <std::floating_point Fp>
inline ParamGate<Fp> ParamRZ(std::uint64_t target,
                             Fp param_coef = 1.,
                             const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamRZGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), param_coef);
}
template <std::floating_point Fp>
inline ParamGate<Fp> ParamPauliRotation(const PauliOperator<Fp>& pauli,
                                        Fp param_coef = 1.,
                                        const std::vector<std::uint64_t>& controls = {}) {
    return internal::ParamGateFactory::create_gate<internal::ParamPauliRotationGateImpl<Fp>>(
        internal::vector_to_mask(controls), pauli, param_coef);
}
template <std::floating_point Fp>
inline ParamGate<Fp> ParamProbablistic(
    const std::vector<Fp>& distribution,
    const std::vector<std::variant<Gate<Fp>, ParamGate<Fp>>>& gate_list) {
    return internal::ParamGateFactory::create_gate<internal::ParamProbablisticGateImpl<Fp>>(
        distribution, gate_list);
}
}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_gate_param_gate_factory(nb::module_& mgate) {
    mgate.def("ParamRX",
              &gate::ParamRX<Fp>,
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general ParamGate class instance of ParamRX.")
                  .arg("target", "int", "tTarget qubit index")
                  .arg("coef", "float", true, "Parameter coefficient")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "ParamRX gate instance")
                  .ex(DocString::Code({">>> gate = ParamRX(0)  # ParamRX gate on qubit 0",
                                       ">>> gate = ParamRX(1, [0])  # Controlled-ParamRX"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("ParamRY",
              &gate::ParamRY<Fp>,
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general ParamGate class instance of ParamRY.")
                  .arg("target", "int", "Target qubit index")
                  .arg("coef", "float", true, "Parameter coefficient")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "ParamRY gate instance")
                  .ex(DocString::Code({">>> gate = ParamRY(0)  # ParamRY gate on qubit 0",
                                       ">>> gate = ParamRY(1, [0])  # Controlled-ParamRY"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("ParamRZ",
              &gate::ParamRZ<Fp>,
              "Generate general ParamGate class instance of ParamRZ.",
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general ParamGate class instance of ParamRZ.")
                  .arg("target", "int", "Target qubit index")
                  .arg("coef", "float", true, "Parameter coefficient")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "ParamRZ gate instance")
                  .ex(DocString::Code({">>> gate = ParamRZ(0)  # ParamRZ gate on qubit 0",
                                       ">>> gate = ParamRZ(1, [0])  # Controlled-ParamRZ"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "ParamPauliRotation",
        &gate::ParamPauliRotation<Fp>,
        "pauli"_a,
        "coef"_a = 1.,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general ParamGate class instance of ParamPauliRotation.")
            .arg("pauli", "PauliOperator", "Pauli operator")
            .arg("coef", "float", true, "Parameter coefficient")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "ParamPauliRotation gate instance")
            .ex(DocString::Code({">>> gate = ParamPauliRotation(PauliOperator(), 0.5)  # Pauli "
                                 "rotation gate with PauliOperator and coefficient 0.5",
                                 ">>> gate = ParamPauliRotation(PauliOperator(), 0.5, [0])  # "
                                 "Controlled-ParamPauliRotation"}))
            .build_as_google_style()
            .c_str());
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
        DocString()
            .desc("Generate general ParamGate class instance of ParamProbablistic.")
            .arg("prob_gate_list",
                 "list[tuple[float, Union[Gate, ParamGate]]]",
                 "List of tuple of probability and gate")
            .ret("Gate", "ParamProbablistic gate instance")
            .ex(DocString::Code({">>> gate = ParamProbablistic([(0.1, X(0)), (0.9, I(0))])  # "
                                 "Probablistic gate with X and I"}))
            .build_as_google_style()
            .c_str());
}
}  // namespace internal
#endif
}  // namespace scaluq
