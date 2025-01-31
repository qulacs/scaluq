#include <scaluq/gate/gate.hpp>

#include "../util/template.hpp"

namespace scaluq {
namespace internal {
<<<<<<< HEAD
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(const StateVector<Prec>& state_vector) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void GateBase<Fp, Sp>::check_qubit_mask_within_bounds(
    const StateVector<Fp, Sp>& state_vector) const {
>>>>>>> set-space
    std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
<<<<<<< HEAD
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(const StateVectorBatched<Prec>& states) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void GateBase<Fp, Sp>::check_qubit_mask_within_bounds(
    const StateVectorBatched<Fp, Sp>& states) const {
>>>>>>> set-space
    std::uint64_t full_mask = (1ULL << states.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVectorBatched& states): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}

<<<<<<< HEAD
template <Precision Prec>
std::string GateBase<Prec>::get_qubit_info_as_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string GateBase<Fp, Sp>::get_qubit_info_as_string(const std::string& indent) const {
>>>>>>> set-space
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

<<<<<<< HEAD
template <Precision Prec>
GateBase<Prec>::GateBase(std::uint64_t target_mask, std::uint64_t control_mask)
=======
FLOAT_AND_SPACE(Fp, Sp)
GateBase<Fp, Sp>::GateBase(std::uint64_t target_mask, std::uint64_t control_mask)
>>>>>>> set-space
    : _target_mask(target_mask), _control_mask(control_mask) {
    if (_target_mask & _control_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::Gate(std::uint64_t target_mask, std::uint64_t control_mask) : Target "
            "and control qubits must not overlap.");
    }
}

<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(GateBase)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(GateBase)
>>>>>>> set-space
}  // namespace internal
}  // namespace scaluq
