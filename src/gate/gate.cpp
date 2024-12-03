#include <scaluq/gate/gate.hpp>

#include "../util/template.hpp"

namespace scaluq {
namespace internal {
FLOAT(Fp)
void GateBase<Fp>::check_qubit_mask_within_bounds(const StateVector<Fp>& state_vector) const {
    std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}

FLOAT(Fp)
std::string GateBase<Fp>::get_qubit_info_as_string(const std::string& indent) const {
    std::ostringstream ss;
    auto targets = target_qubit_list();
    auto controls = control_qubit_list();
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

FLOAT(Fp)
GateBase<Fp>::GateBase(std::uint64_t target_mask, std::uint64_t control_mask)
    : _target_mask(target_mask), _control_mask(control_mask) {
    if (_target_mask & _control_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::Gate(std::uint64_t target_mask, std::uint64_t control_mask) : Target "
            "and control qubits must not overlap.");
    }
}

FLOAT_DECLARE_CLASS(GateBase)
}  // namespace internal
}  // namespace scaluq
