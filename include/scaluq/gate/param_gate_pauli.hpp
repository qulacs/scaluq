#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec>
class ParamPauliRotationGateImpl : public ParamGateBase<Prec> {
    const PauliOperator<Prec> _pauli;

public:
    ParamPauliRotationGateImpl(std::uint64_t control_mask,
                               const PauliOperator<Prec>& pauli,
                               Float<Prec> param_coef = 1)
        : ParamGateBase<Prec>(
              vector_to_mask<false>(pauli.target_qubit_list()), control_mask, param_coef),
          _pauli(pauli) {}

    PauliOperator<Prec> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamPauliRotationGateImpl<Prec>>(
            this->_control_mask, _pauli, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;
    void update_quantum_state(StateVector<Prec>& state_vector, double param) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states,
                              std::vector<double> params) const override;
    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamPauliRotation"},
                 {"control", this->control_qubit_list()},
                 {"pauli", this->pauli()},
                 {"param_coef", this->param_coef()}};
    }
};
}  // namespace internal

template <Precision Prec>
using ParamPauliRotationGate = internal::ParamGatePtr<internal::ParamPauliRotationGateImpl<Prec>>;

namespace internal {
#define DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_PRECISION(Prec)                                \
    template <>                                                                                   \
    inline std::shared_ptr<const ParamPauliRotationGateImpl<Prec>> get_from_json(const Json& j) { \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                        \
        auto pauli = j.at("pauli").get<PauliOperator<Prec>>();                                    \
        auto param_coef = j.at("param_coef").get<double>();                                       \
        return std::make_shared<const ParamPauliRotationGateImpl<Prec>>(                          \
            vector_to_mask(controls), pauli, static_cast<Float<Prec>>(param_coef));               \
    }

#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_PRECISION

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_param_gate_pauli_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamPauliRotationGate,
        Prec,
        "Specific class of parametric multi-qubit pauli-rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}P}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
