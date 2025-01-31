#include <scaluq/gate/gate_pauli.hpp>

#include "../operator/apply_pauli.hpp"
#include "../util/math.hpp"
#include "../util/template.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
template <Precision Prec>
void PauliGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void PauliGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
>>>>>>> set-space
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli(this->_control_mask,
                bit_flip_mask,
                phase_flip_mask,
                Complex<Prec>(_pauli.coef()),
                state_vector);
}
<<<<<<< HEAD
template <Precision Prec>
void PauliGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void PauliGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
>>>>>>> set-space
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli(
        this->_control_mask, bit_flip_mask, phase_flip_mask, Complex<Prec>(_pauli.coef()), states);
}
<<<<<<< HEAD
template <Precision Prec>
std::string PauliGateImpl<Prec>::to_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string PauliGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
>>>>>>> set-space
    std::ostringstream ss;
    auto controls = this->control_qubit_list();
    ss << indent << "Gate Type: Pauli\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Pauli Operator: \"" << _pauli.get_pauli_string() << "\"";
    return ss.str();
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(PauliGateImpl)

template <Precision Prec>
ComplexMatrix PauliRotationGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
    StdComplex true_angle = static_cast<double>(_angle) * _pauli.coef();
    StdComplex half_angle = true_angle / 2.;
    StdComplex imag_unit(0, 1);
    mat = std::cos(-half_angle) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
          imag_unit * std::sin(-half_angle) * mat;
    return mat;
}
template <Precision Prec>
void PauliRotationGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector) const {
=======
FLOAT_AND_SPACE_DECLARE_CLASS(PauliGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> PauliRotationGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat = this->_pauli.get_matrix_ignoring_coef();
    Complex<Fp> true_angle = _angle * _pauli.coef();
    StdComplex<Fp> imag_unit(0, 1);
    mat = (StdComplex<Fp>)Kokkos::cos(-true_angle / 2) *
              internal::ComplexMatrix<Fp>::Identity(mat.rows(), mat.cols()) +
          imag_unit * (StdComplex<Fp>)Kokkos::sin(-true_angle / 2) * mat;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void PauliRotationGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
>>>>>>> set-space
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         _angle,
                         state_vector);
}
<<<<<<< HEAD
template <Precision Prec>
void PauliRotationGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void PauliRotationGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
>>>>>>> set-space
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         _angle,
                         states);
}
<<<<<<< HEAD
template <Precision Prec>
std::string PauliRotationGateImpl<Prec>::to_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string PauliRotationGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
>>>>>>> set-space
    std::ostringstream ss;
    auto controls = this->control_qubit_list();
    ss << indent << "Gate Type: PauliRotation\n";
    ss << indent << "  Angle: " << static_cast<double>(_angle) << "\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Pauli Operator: \"" << _pauli.get_pauli_string() << "\"";
    return ss.str();
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(PauliRotationGateImpl)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(PauliRotationGateImpl)
>>>>>>> set-space
}  // namespace scaluq::internal
