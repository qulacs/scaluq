#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec, ExecutionSpace Space>
class ParamPauliRotationGateImpl : public ParamGateBase<Prec, Space> {
    const PauliOperator<Prec, Space> _pauli;

public:
    ParamPauliRotationGateImpl(std::uint64_t control_mask,
                               std::uint64_t control_value_mask,
                               const PauliOperator<Prec, Space>& pauli,
                               Float<Prec> param_coef = 1)
        : ParamGateBase<Prec, Space>(vector_to_mask<false>(pauli.target_qubit_list()),
                                     control_mask,
                                     control_value_mask,
                                     param_coef),
          _pauli(pauli) {}

    PauliOperator<Prec, Space> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const ParamGateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const ParamPauliRotationGateImpl<Prec, Space>>(
            this->_control_mask, this->_control_value_mask, _pauli, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;
    void update_quantum_state(StateVector<Prec, Space>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states,
                              std::vector<double> params) const override;
    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamPauliRotation"},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"pauli", this->pauli()},
                 {"param_coef", this->param_coef()}};
    }
};
}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using ParamPauliRotationGate =
    internal::ParamGatePtr<internal::ParamPauliRotationGateImpl<Prec, Space>>;

namespace internal {
#define DECLARE_GET_FROM_JSON(Prec, Space)                                               \
    template <>                                                                          \
    inline std::shared_ptr<const ParamPauliRotationGateImpl<Prec, Space>> get_from_json( \
        const Json& j) {                                                                 \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();               \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();   \
        return std::make_shared<const ParamPauliRotationGateImpl<Prec, Space>>(          \
            vector_to_mask(controls),                                                    \
            vector_to_mask(controls, control_values),                                    \
            j.at("pauli").get<PauliOperator<Prec, Space>>(),                             \
            static_cast<Float<Prec>>(j.at("param_coef").get<double>()));                 \
    }

#define INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Prec)       \
    DECLARE_GET_FROM_JSON(Prec, ExecutionSpace::Default) \
    DECLARE_GET_FROM_JSON(Prec, ExecutionSpace::Host)

#ifdef SCALUQ_BFLOAT16
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::BF16)
#endif
#ifdef SCALUQ_FLOAT16
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::F64)
#endif

#undef DECLARE_GET_FROM_JSON
#undef INSTANTIATE_GET_FROM_JSON_EACH_SPACE

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_param_gate_pauli_hpp(nb::module_& m,
                                    nb::class_<ParamGate<Prec, Space>>& param_gate_base_def) {
    DEF_PARAM_GATE(ParamPauliRotationGate,
                   Prec,
                   Space,
                   "Specific class of parametric multi-qubit pauli-rotation gate, represented as "
                   "$e^{-i\\frac{\\theta}{2}P}$. `theta` is given as `param * param_coef`.",
                   param_gate_base_def);
}
}  // namespace internal
#endif
}  // namespace scaluq
