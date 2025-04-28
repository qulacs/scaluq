#include <scaluq/gate/param_gate.hpp>

#include "../prec_space.hpp"

namespace scaluq::internal {
template <Precision Prec, ExecutionSpace Space>
void ParamGateBase<Prec, Space>::check_qubit_mask_within_bounds(
    const StateVector<Prec, Space>& state_vector) const {
    std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: ParamGate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
template <Precision Prec, ExecutionSpace Space>
void ParamGateBase<Prec, Space>::check_qubit_mask_within_bounds(
    const StateVectorBatched<Prec, Space>& states) const {
    std::uint64_t full_mask = (1ULL << states.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: ParamGate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
template <Precision Prec, ExecutionSpace Space>
std::string ParamGateBase<Prec, Space>::get_qubit_info_as_string(const std::string& indent) const {
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
template <Precision Prec, ExecutionSpace Space>
ParamGateBase<Prec, Space>::ParamGateBase(std::uint64_t target_mask,
                                          std::uint64_t control_mask,
                                          std::uint64_t control_value_mask,
                                          Float<Prec> param_coef)
    : _target_mask(target_mask),
      _control_mask(control_mask),
      _control_value_mask(control_value_mask),
      _pcoef(param_coef) {
    if (_target_mask & _control_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: ParamGate::ParamGate(std::uint64_t target_mask, std::uint64_t "
            "control_mask) : Target and control qubits must not overlap.");
    }
}
template class ParamGateBase<Prec, Space>;
}  // namespace scaluq::internal
