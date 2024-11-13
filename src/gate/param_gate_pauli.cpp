#include <scaluq/gate/param_gate_pauli.hpp>

#include "../operator/apply_pauli.hpp"
#include "../util/template.hpp"

namespace scaluq::internal {
FLOAT(Fp)
ComplexMatrix<Fp> ParamPauliRotationGateImpl<Fp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    Complex<Fp> true_angle = angle * this->_pauli.coef();
    internal::ComplexMatrix<Fp> mat = this->_pauli.get_matrix_ignoring_coef();
    StdComplex<Fp> imag_unit(0, 1);
    mat = (StdComplex<Fp>)Kokkos::cos(-true_angle / 2) *
              internal::ComplexMatrix<Fp>::Identity(mat.rows(), mat.cols()) +
          imag_unit * (StdComplex<Fp>)Kokkos::sin(-true_angle / 2) * mat;
    return mat;
}
FLOAT(Fp)
void ParamPauliRotationGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector,
                                                          Fp param) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         _pauli.coef(),
                         this->_pcoef * param,
                         state_vector);
}
FLOAT(Fp)
std::string ParamPauliRotationGateImpl<Fp>::to_string(const std::string& indent) const {
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
FLOAT_DECLARE_CLASS(ParamPauliRotationGateImpl)
}  // namespace scaluq::internal
