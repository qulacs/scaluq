#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
template <std::floating_point Fp>
class ParamPauliRotationGateImpl : public ParamGateBase<Fp> {
    const PauliOperator<Fp> _pauli;

public:
    ParamPauliRotationGateImpl(std::uint64_t control_mask,
                               const PauliOperator<Fp>& pauli,
                               Fp param_coef = 1.)
        : ParamGateBase<Fp>(
              vector_to_mask<false>(pauli.target_qubit_list()), control_mask, param_coef),
          _pauli(pauli) {}

    PauliOperator<Fp> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const ParamGateBase<Fp>> get_inverse() const override {
        return std::make_shared<const ParamPauliRotationGateImpl<Fp>>(
            this->_control_mask, _pauli, -this->_pcoef);
    }
    internal::ComplexMatrix<Fp> get_matrix(Fp param) const override;
    void update_quantum_state(StateVector<Fp>& state_vector, Fp param) const override;
    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "ParamPauliRotation"},
                 {"control", this->control_qubit_list()},
                 {"pauli", this->pauli()},
                 {"param_coef", this->param_coef()}};
    }
};
}  // namespace internal

template <std::floating_point Fp>
using ParamPauliRotationGate = internal::ParamGatePtr<internal::ParamPauliRotationGateImpl<Fp>>;

namespace internal {
#define DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_TYPE(Type)                                     \
    template <>                                                                                   \
    inline std::shared_ptr<const ParamPauliRotationGateImpl<Type>> get_from_json(const Json& j) { \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                        \
        auto pauli = j.at("pauli").get<PauliOperator<Type>>();                                    \
        auto param_coef = j.at("param_coef").get<Type>();                                         \
        return std::make_shared<const ParamPauliRotationGateImpl<Type>>(                          \
            vector_to_mask(controls), pauli, param_coef);                                         \
    }

DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_PARAM_PAULIGATE_WITH_TYPE

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_gate_param_gate_pauli_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamPauliRotationGate,
        Fp,
        "Specific class of parametric multi-qubit pauli-rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}P}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
