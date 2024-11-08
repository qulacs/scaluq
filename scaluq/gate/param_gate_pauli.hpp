#pragma once

#include <vector>

#include "../operator/apply_pauli.hpp"
#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "param_gate.hpp"

namespace scaluq {
namespace internal {
class ParamPauliRotationGateImpl : public ParamGateBase {
    const PauliOperator _pauli;

public:
    ParamPauliRotationGateImpl(std::uint64_t control_mask,
                               const PauliOperator& pauli,
                               double param_coef = 1.)
        : ParamGateBase(vector_to_mask<false>(pauli.target_qubit_list()), control_mask, param_coef),
          _pauli(pauli) {}

    PauliOperator pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    ParamGate get_inverse() const override {
        return std::make_shared<const ParamPauliRotationGateImpl>(_control_mask, _pauli, -_pcoef);
    }
    internal::ComplexMatrix get_matrix(double param) const override {
        double angle = _pcoef * param;
        Complex true_angle = angle * this->_pauli.coef();
        internal::ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
        StdComplex imag_unit(0, 1);
        mat = (StdComplex)Kokkos::cos(-true_angle / 2) *
                  internal::ComplexMatrix::Identity(mat.rows(), mat.cols()) +
              imag_unit * (StdComplex)Kokkos::sin(-true_angle / 2) * mat;
        return mat;
    }
    void update_quantum_state(StateVector& state_vector, double param) const override {
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
        apply_pauli_rotation(_control_mask,
                             bit_flip_mask,
                             phase_flip_mask,
                             _pauli.coef(),
                             _pcoef * param,
                             state_vector);
    }
    void update_quantum_state(StateVectorBatched& states,
                              std::vector<double> params) const override {
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
        apply_pauli_rotation(
            _control_mask, bit_flip_mask, phase_flip_mask, _pauli.coef(), _pcoef, params, states);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        auto controls = control_qubit_list();
        ss << indent << "Gate Type: ParamPauliRotation\n";
        ss << indent << "  Control Qubits: {";
        for (std::uint32_t i = 0; i < controls.size(); ++i)
            ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
        ss << "}\n";
        ss << indent << "  Pauli Operator: \"" << _pauli.get_pauli_string() << "\"";
        return ss.str();
    }
};
}  // namespace internal

using ParamPauliRotationGate = internal::ParamGatePtr<internal::ParamPauliRotationGateImpl>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_param_gate_pauli_hpp(nb::module_& m) {
    DEF_PARAM_GATE(
        ParamPauliRotationGate,
        "Specific class of parametric multi-qubit pauli-rotation gate, represented as "
        "$e^{-i\\frac{\\mathrm{angle}}{2}P}$. `angle` is given as `param * param_coef`.");
}
}  // namespace internal
#endif
}  // namespace scaluq
