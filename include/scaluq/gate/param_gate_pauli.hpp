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
                               std::uint64_t control_value_mask,
                               const PauliOperator<Prec>& pauli,
                               Float<Prec> param_coef = 1)
        : ParamGateBase<Prec>(vector_to_mask<false>(pauli.target_qubit_list()),
                              control_mask,
                              control_value_mask,
                              param_coef),
          _pauli(pauli) {}

    PauliOperator<Prec> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const ParamGateBase<Prec>> get_inverse() const override {
        return std::make_shared<const ParamPauliRotationGateImpl<Prec>>(
            this->_control_mask, this->_control_value_mask, _pauli, -this->_pcoef);
    }
    ComplexMatrix get_matrix(double param) const override;
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Host>& states,
                              std::vector<double> params) const override;
    void update_quantum_state(StateVector<Prec, ExecutionSpace::HostSerial>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
                              std::vector<double> params) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(StateVector<Prec, ExecutionSpace::Default>& state_vector,
                              double param) const override;
    void update_quantum_state(StateVectorBatched<Prec, ExecutionSpace::Default>& states,
                              std::vector<double> params) const override;
#endif  // SCALUQ_USE_CUDA
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

template <Precision Prec>
using ParamPauliRotationGate = internal::ParamGatePtr<internal::ParamPauliRotationGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_param_gate_pauli_hpp(nb::module_& m,
                                    nb::class_<ParamGate<Prec>>& param_gate_base_def) {
    bind_specific_param_gate<ParamPauliRotationGate<Prec>, Prec>(
        m,
        param_gate_base_def,
        "ParamPauliRotationGate",
        "Parametric multi-qubit pauli-rotation gate, represented as $e^{-i\\frac{\\theta}{2}P}$. "
        "`theta` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
