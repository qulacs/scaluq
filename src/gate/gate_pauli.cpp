#include <scaluq/gate/gate_pauli.hpp>

#include "../operator/apply_pauli.hpp"
#include "../prec_space.hpp"
#include "../util/math.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec, ExecutionSpace Space>
void PauliGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli(this->_control_mask,
                this->_control_value_mask,
                bit_flip_mask,
                phase_flip_mask,
                Complex<Prec>(_pauli.coef()),
                state_vector);
}
template <Precision Prec, ExecutionSpace Space>
void PauliGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli(this->_control_mask,
                this->_control_value_mask,
                bit_flip_mask,
                phase_flip_mask,
                Complex<Prec>(_pauli.coef()),
                states);
}
template <Precision Prec, ExecutionSpace Space>
std::string PauliGateImpl<Prec, Space>::to_string(const std::string& indent) const {
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
template class PauliGateImpl<Prec, Space>;

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix PauliRotationGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
    StdComplex true_angle = static_cast<double>(_angle) * _pauli.coef();
    StdComplex half_angle = true_angle / 2.;
    StdComplex imag_unit(0, 1);
    mat = std::cos(-half_angle) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
          imag_unit * std::sin(-half_angle) * mat;
    return mat;
}
template <Precision Prec, ExecutionSpace Space>
void PauliRotationGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         _angle,
                         state_vector);
}
template <Precision Prec, ExecutionSpace Space>
void PauliRotationGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         _angle,
                         states);
}
template <Precision Prec, ExecutionSpace Space>
std::string PauliRotationGateImpl<Prec, Space>::to_string(const std::string& indent) const {
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
template class PauliRotationGateImpl<Prec, Space>;
}  // namespace scaluq::internal
