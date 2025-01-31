#include <scaluq/gate/param_gate.hpp>

#include "../util/template.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
template <Precision Prec>
void ParamGateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVector<Prec>& state_vector) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ParamGateBase<Fp, Sp>::check_qubit_mask_within_bounds(
    const StateVector<Fp, Sp>& state_vector) const {
>>>>>>> set-space
    std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: ParamGate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
<<<<<<< HEAD
template <Precision Prec>
void ParamGateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVectorBatched<Prec>& states) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ParamGateBase<Fp, Sp>::check_qubit_mask_within_bounds(
    const StateVectorBatched<Fp, Sp>& states) const {
>>>>>>> set-space
    std::uint64_t full_mask = (1ULL << states.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: ParamGate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
<<<<<<< HEAD
template <Precision Prec>
std::string ParamGateBase<Prec>::get_qubit_info_as_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamGateBase<Fp, Sp>::get_qubit_info_as_string(const std::string& indent) const {
>>>>>>> set-space
    std::ostringstream ss;
    auto targets = target_qubit_list();
    auto controls = control_qubit_list();
    ss << indent << "  Parameter Coefficient: " << static_cast<double>(_pcoef) << "\n";
    ss << indent << "  Target Qubits: {";
    for (std::uint32_t i = 0; i < targets.size(); ++i)
        ss << targets[i] << (i == targets.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}";
    return ss.str();
}
<<<<<<< HEAD
template <Precision Prec>
ParamGateBase<Prec>::ParamGateBase(std::uint64_t target_mask,
                                   std::uint64_t control_mask,
                                   Float<Prec> param_coef)
=======
FLOAT_AND_SPACE(Fp, Sp)
ParamGateBase<Fp, Sp>::ParamGateBase(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     Fp param_coef)
>>>>>>> set-space
    : _target_mask(target_mask), _control_mask(control_mask), _pcoef(param_coef) {
    if (_target_mask & _control_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: ParamGate::ParamGate(std::uint64_t target_mask, std::uint64_t "
            "control_mask) : Target and control qubits must not overlap.");
    }
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(ParamGateBase)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(ParamGateBase)
>>>>>>> set-space
}  // namespace scaluq::internal
