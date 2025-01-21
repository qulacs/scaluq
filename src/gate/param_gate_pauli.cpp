#include <scaluq/gate/param_gate_pauli.hpp>

#include "../operator/apply_pauli.hpp"
#include "../util/template.hpp"

namespace scaluq::internal {
FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> ParamPauliRotationGateImpl<Fp, Sp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    Complex<Fp> true_angle = angle * this->_pauli.coef();
    internal::ComplexMatrix<Fp> mat = this->_pauli.get_matrix_ignoring_coef();
    StdComplex<Fp> imag_unit(0, 1);
    mat = (StdComplex<Fp>)Kokkos::cos(-true_angle / 2) *
              internal::ComplexMatrix<Fp>::Identity(mat.rows(), mat.cols()) +
          imag_unit * (StdComplex<Fp>)Kokkos::sin(-true_angle / 2) * mat;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamPauliRotationGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector,
                                                              Fp param) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         _pauli.coef(),
                         this->_pcoef * param,
                         state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamPauliRotationGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                              std::vector<Fp> params) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         _pauli.coef(),
                         this->_pcoef,
                         params,
                         states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamPauliRotationGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    auto controls = this->control_qubit_list();
    ss << indent << "Gate Type: ParamPauliRotation\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Pauli Operator: \"" << _pauli.get_pauli_string() << "\"";
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(ParamPauliRotationGateImpl)
}  // namespace scaluq::internal
