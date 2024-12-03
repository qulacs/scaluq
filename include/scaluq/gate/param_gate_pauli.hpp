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
    void update_quantum_state(StateVectorBatched<Fp>& states,
                              std::vector<Fp> params) const override;
    std::string to_string(const std::string& indent) const override;
};
}  // namespace internal

template <std::floating_point Fp>
using ParamPauliRotationGate = internal::ParamGatePtr<internal::ParamPauliRotationGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_param_gate_pauli_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamPauliRotationGate,
        double,
        "Specific class of parametric multi-qubit pauli-rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}P}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
