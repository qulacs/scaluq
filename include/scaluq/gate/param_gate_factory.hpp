#pragma once

#include "../util/utility.hpp"
#include "param_gate_pauli.hpp"
#include "param_gate_probabilistic.hpp"
#include "param_gate_standard.hpp"

namespace scaluq {
namespace internal {
class ParamGateFactory {
public:
    template <ParamGateImpl T, typename... Args>
    static ParamGate<T::Prec, T::Space> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {
template <Precision Prec, ExecutionSpace Space>
inline ParamGate<Prec, Space> ParamRX(std::uint64_t target,
                                      double param_coef = 1.,
                                      const std::vector<std::uint64_t>& controls = {},
                                      std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::ParamGateFactory::create_gate<internal::ParamRXGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec, ExecutionSpace Space>
inline ParamGate<Prec, Space> ParamRY(std::uint64_t target,
                                      double param_coef = 1.,
                                      const std::vector<std::uint64_t>& controls = {},
                                      std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::ParamGateFactory::create_gate<internal::ParamRYGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec, ExecutionSpace Space>
inline ParamGate<Prec, Space> ParamRZ(std::uint64_t target,
                                      double param_coef = 1.,
                                      const std::vector<std::uint64_t>& controls = {},
                                      std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::ParamGateFactory::create_gate<internal::ParamRZGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec, ExecutionSpace Space>
inline ParamGate<Prec, Space> ParamPauliRotation(const PauliOperator<Prec, Space>& pauli,
                                                 double param_coef = 1.,
                                                 const std::vector<std::uint64_t>& controls = {},
                                                 std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::ParamGateFactory::create_gate<
        internal::ParamPauliRotationGateImpl<Prec, Space>>(
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        pauli,
        static_cast<internal::Float<Prec>>(param_coef));
}
template <Precision Prec, ExecutionSpace Space>
inline ParamGate<Prec, Space> ParamProbabilistic(
    const std::vector<double>& distribution,
    const std::vector<std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>>& gate_list) {
    return internal::ParamGateFactory::create_gate<
        internal::ParamProbabilisticGateImpl<Prec, Space>>(distribution, gate_list);
}
}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_param_gate_factory(nb::module_& mgate) {
    mgate.def(
        "ParamRX",
        &gate::ParamRX<Prec, Space>,
        "target"_a,
        "coef"_a = 1.,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.ParamGate` class instance of "
                  ":class:`~scaluq.f64.ParamRXGate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.ParamRXGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("coef", "float", true, "Parameter coefficient")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .ret("ParamGate", "ParamRX gate instance")
            .ex(DocString::Code({">>> gate = ParamRX(0)  # ParamRX gate on qubit 0",
                                 ">>> gate = ParamRX(1, [0])  # Controlled-ParamRX"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "ParamRY",
        &gate::ParamRY<Prec, Space>,
        "target"_a,
        "coef"_a = 1.,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.ParamGate` class instance of "
                  ":class:`~scaluq.f64.ParamRYGate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.ParamRYGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("coef", "float", true, "Parameter coefficient")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .ret("ParamGate", "ParamRY gate instance")
            .ex(DocString::Code({">>> gate = ParamRY(0)  # ParamRY gate on qubit 0",
                                 ">>> gate = ParamRY(1, [0])  # Controlled-ParamRY"}))
            .build_as_google_style()
            .c_str());
    mgate.def("ParamRZ",
              &gate::ParamRZ<Prec, Space>,
              "target"_a,
              "coef"_a = 1.,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.ParamGate` class instance of "
                        ":class:`~scaluq.f64.ParamRZGate`.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.f64.ParamRZGate` class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("coef", "float", true, "Parameter coefficient")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .ret("ParamGate", "ParamRZ gate instance")
                  .ex(DocString::Code({">>> gate = ParamRZ(0)  # ParamRZ gate on qubit 0",
                                       ">>> gate = ParamRZ(1, [0])  # Controlled-ParamRZ"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "ParamPauliRotation",
        &gate::ParamPauliRotation<Prec, Space>,
        "pauli"_a,
        "coef"_a = 1.,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.ParamGate` class instance of "
                  ":class:`~scaluq.f64.ParamPauliRotationGate`.")
            .note("If you need to use functions specific to the "
                  ":class:`~scaluq.f64.ParamPauliRotationGate` "
                  "class, please downcast it.")
            .arg("pauli", "PauliOperator", "Pauli operator")
            .arg("coef", "float", true, "Parameter coefficient")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .ret("ParamGate", "ParamPauliRotation gate instance")
            .ex(DocString::Code({">>> gate = ParamPauliRotation(PauliOperator(), 0.5)  # Pauli "
                                 "rotation gate with PauliOperator and coefficient 0.5",
                                 ">>> gate = ParamPauliRotation(PauliOperator(), 0.5, [0])  # "
                                 "Controlled-ParamPauliRotation"}))
            .build_as_google_style()
            .c_str());
    mgate.def("ParamProbabilistic",
              &gate::ParamProbabilistic<Prec, Space>,
              "distribution"_a,
              "gate_list"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.ParamGate` class instance of "
                        ":class:`~scaluq.f64.ParamProbabilisticGate`.")
                  .arg("distribution", "list[float]", "List of probability")
                  .arg("gate_list", "list[Union[Gate, ParamGate]]", "List of gates")
                  .ret("ParamGate", "ParamProbabilistic gate instance")
                  .ex(DocString::Code(
                      {">>> gate = ParamProbabilistic([0.1, 0.9], [X(0), ParamRX(0, 0.5)])  # "
                       "probabilistic gate with X and ParamRX"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "ParamProbabilistic",
        [](const std::vector<
            std::pair<double, std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>>>&
               prob_gate_list) {
            std::vector<double> distribution;
            std::vector<std::variant<Gate<Prec, Space>, ParamGate<Prec, Space>>> gate_list;
            distribution.reserve(prob_gate_list.size());
            gate_list.reserve(prob_gate_list.size());
            for (const auto& [prob, gate] : prob_gate_list) {
                distribution.push_back(prob);
                gate_list.push_back(gate);
            }
            return gate::ParamProbabilistic<Prec, Space>(distribution, gate_list);
        },
        "prob_gate_list"_a,
        DocString()
            .desc("Generate general :class:`~scaluq.f64.ParamGate` class instance of "
                  ":class:`~scaluq.f64.ParamProbabilisticGate`.")
            .arg("prob_gate_list",
                 "list[tuple[float, Union[Gate, ParamGate]]]",
                 "List of tuple of probability and gate")
            .ret("ParamGate", "ParamProbabilistic gate instance")
            .ex(DocString::Code({">>> gate = ParamProbabilistic([(0.1, X(0)), (0.9, I(0))])  # "
                                 "probabilistic gate with X and I"}))
            .build_as_google_style()
            .c_str());
}
}  // namespace internal
#endif
}  // namespace scaluq
