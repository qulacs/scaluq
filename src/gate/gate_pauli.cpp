#include <scaluq/gate/gate_pauli.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec>
std::string PauliGateImpl<Prec>::to_string(const std::string& indent) const {
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
#define DEFINE_PAULI_GATE_UPDATE(Class, Space)                                               \
    template <Precision Prec>                                                                \
    void PauliGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();         \
        apply_pauli(this->_control_mask,                                                     \
                    this->_control_value_mask,                                               \
                    bit_flip_mask,                                                           \
                    phase_flip_mask,                                                         \
                    Complex<Prec>(_pauli.coef()),                                            \
                    state_vector);                                                           \
    }
DEFINE_PAULI_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_PAULI_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_PAULI_GATE_UPDATE(StateVector, ExecutionSpace::HostSerialSpace)
DEFINE_PAULI_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerialSpace)
#ifdef SCALUQ_USE_CUDA
DEFINE_PAULI_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_PAULI_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_PAULI_GATE_UPDATE
template class PauliGateImpl<Prec>;

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
std::string PauliRotationGateImpl<Prec>::to_string(const std::string& indent) const {
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
#define DEFINE_PAULI_ROTATION_GATE_UPDATE(Class, Space)                                      \
    template <Precision Prec>                                                                \
    void PauliRotationGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) \
        const {                                                                              \
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();         \
        apply_pauli_rotation(this->_control_mask,                                            \
                             this->_control_value_mask,                                      \
                             bit_flip_mask,                                                  \
                             phase_flip_mask,                                                \
                             Complex<Prec>(_pauli.coef()),                                   \
                             _angle,                                                         \
                             state_vector);                                                  \
    }
DEFINE_PAULI_ROTATION_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_PAULI_ROTATION_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_PAULI_ROTATION_GATE_UPDATE(StateVector, ExecutionSpace::HostSerialSpace)
DEFINE_PAULI_ROTATION_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerialSpace)
#ifdef SCALUQ_USE_CUDA
DEFINE_PAULI_ROTATION_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_PAULI_ROTATION_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_PAULI_ROTATION_GATE_UPDATE
template class PauliRotationGateImpl<Prec>;

template <Precision Prec>
std::shared_ptr<const PauliGateImpl<Prec>> GetGateFromJson<PauliGateImpl<Prec>>::get(
    const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const PauliGateImpl<Prec>>(
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        j.at("pauli").get<PauliOperator<Prec>>());
}
template class GetGateFromJson<PauliGateImpl<Prec>>;
template <>
std::shared_ptr<const PauliRotationGateImpl<Prec>>
GetGateFromJson<PauliRotationGateImpl<Prec>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const PauliRotationGateImpl<Prec>>(
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        j.at("pauli").get<PauliOperator<Prec>>(),
        static_cast<Float<Prec>>(j.at("angle").get<double>()));
}
template class GetGateFromJson<PauliRotationGateImpl<Prec>>;
}  // namespace scaluq::internal
