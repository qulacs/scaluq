#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/prec_space.hpp>

namespace scaluq::internal {
template <Precision Prec>
GateType get_runtime_gate_type(const GateBase<Prec>& gate) {
    const auto& type = typeid(gate);
    if (type == typeid(IGateImpl<Prec>)) return GateType::I;
    if (type == typeid(GlobalPhaseGateImpl<Prec>)) return GateType::GlobalPhase;
    if (type == typeid(XGateImpl<Prec>)) return GateType::X;
    if (type == typeid(YGateImpl<Prec>)) return GateType::Y;
    if (type == typeid(ZGateImpl<Prec>)) return GateType::Z;
    if (type == typeid(HGateImpl<Prec>)) return GateType::H;
    if (type == typeid(SGateImpl<Prec>)) return GateType::S;
    if (type == typeid(SdagGateImpl<Prec>)) return GateType::Sdag;
    if (type == typeid(TGateImpl<Prec>)) return GateType::T;
    if (type == typeid(TdagGateImpl<Prec>)) return GateType::Tdag;
    if (type == typeid(SqrtXGateImpl<Prec>)) return GateType::SqrtX;
    if (type == typeid(SqrtXdagGateImpl<Prec>)) return GateType::SqrtXdag;
    if (type == typeid(SqrtYGateImpl<Prec>)) return GateType::SqrtY;
    if (type == typeid(SqrtYdagGateImpl<Prec>)) return GateType::SqrtYdag;
    if (type == typeid(P0GateImpl<Prec>)) return GateType::P0;
    if (type == typeid(P1GateImpl<Prec>)) return GateType::P1;
    if (type == typeid(MeasurementGateImpl<Prec>)) return GateType::Measurement;
    if (type == typeid(RXGateImpl<Prec>)) return GateType::RX;
    if (type == typeid(RYGateImpl<Prec>)) return GateType::RY;
    if (type == typeid(RZGateImpl<Prec>)) return GateType::RZ;
    if (type == typeid(U1GateImpl<Prec>)) return GateType::U1;
    if (type == typeid(U2GateImpl<Prec>)) return GateType::U2;
    if (type == typeid(U3GateImpl<Prec>)) return GateType::U3;
    if (type == typeid(SwapGateImpl<Prec>)) return GateType::Swap;
    if (type == typeid(EcrGateImpl<Prec>)) return GateType::Ecr;
    if (type == typeid(PermutationGateImpl<Prec>)) return GateType::Permutation;
    if (type == typeid(PauliGateImpl<Prec>)) return GateType::Pauli;
    if (type == typeid(PauliRotationGateImpl<Prec>)) return GateType::PauliRotation;
    if (type == typeid(ProbabilisticGateImpl<Prec>)) return GateType::Probabilistic;
    if (type == typeid(DenseMatrixGateImpl<Prec, ExecutionSpace::Host>))
        return GateType::DenseMatrix;
    if (type == typeid(DenseMatrixGateImpl<Prec, ExecutionSpace::HostSerial>))
        return GateType::DenseMatrix;
#ifdef SCALUQ_USE_DEVICE
    if (type == typeid(DenseMatrixGateImpl<Prec, ExecutionSpace::Default>))
        return GateType::DenseMatrix;
#endif
    if (type == typeid(SparseMatrixGateImpl<Prec, ExecutionSpace::Host>))
        return GateType::SparseMatrix;
    if (type == typeid(SparseMatrixGateImpl<Prec, ExecutionSpace::HostSerial>))
        return GateType::SparseMatrix;
#ifdef SCALUQ_USE_DEVICE
    if (type == typeid(SparseMatrixGateImpl<Prec, ExecutionSpace::Default>))
        return GateType::SparseMatrix;
#endif
    return GateType::Unknown;
}
template GateType get_runtime_gate_type<Prec>(const GateBase<Prec>& gate);

template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVector<Prec, ExecutionSpace::Host>& state_vector) const {
    std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVectorBatched<Prec, ExecutionSpace::Host>& states) const {
    std::uint64_t full_mask = (1ULL << states.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVectorBatched& states): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVector<Prec, ExecutionSpace::HostSerial>& state_vector) const {
    std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states) const {
    std::uint64_t full_mask = (1ULL << states.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVectorBatched& states): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
#ifdef SCALUQ_USE_DEVICE
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVector<Prec, ExecutionSpace::Default>& state_vector) const {
    std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVector& state): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const StateVectorBatched<Prec, ExecutionSpace::Default>& states) const {
    std::uint64_t full_mask = (1ULL << states.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(StateVectorBatched& states): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
#endif  // SCALUQ_USE_DEVICE
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const DensityMatrix<Prec, ExecutionSpace::Host>& dm) const {
    std::uint64_t full_mask = (1ULL << dm.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(DensityMatrix& dm): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const DensityMatrix<Prec, ExecutionSpace::HostSerial>& dm) const {
    std::uint64_t full_mask = (1ULL << dm.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(DensityMatrix& dm): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
#ifdef SCALUQ_USE_DEVICE
template <Precision Prec>
void GateBase<Prec>::check_qubit_mask_within_bounds(
    const DensityMatrix<Prec, ExecutionSpace::Default>& dm) const {
    std::uint64_t full_mask = (1ULL << dm.n_qubits()) - 1;
    if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::update_quantum_state(DensityMatrix& dm): "
            "Target/Control qubit exceeds the number of qubits in the system.");
    }
}
#endif  // SCALUQ_USE_DEVICE

template <Precision Prec>
std::string GateBase<Prec>::get_qubit_info_as_string(const std::string& indent) const {
    std::ostringstream ss;
    auto targets = target_qubit_list();
    auto controls = control_qubit_list();
    auto control_values = control_value_list();
    ss << indent << "  Target Qubits: {";
    for (std::uint32_t i = 0; i < targets.size(); ++i)
        ss << targets[i] << (i == targets.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Control Value: {";
    for (std::uint32_t i = 0; i < control_values.size(); ++i)
        ss << control_values[i] << (i == control_values.size() - 1 ? "" : ", ");
    ss << "}";
    return ss.str();
}

template <Precision Prec>
GateBase<Prec>::GateBase(std::uint64_t target_mask,
                         std::uint64_t control_mask,
                         std::uint64_t control_value_mask)
    : _target_mask(target_mask),
      _control_mask(control_mask),
      _control_value_mask(control_value_mask) {
    if (_target_mask & _control_mask) [[unlikely]] {
        throw std::runtime_error(
            "Error: Gate::Gate(std::uint64_t target_mask, std::uint64_t control_mask) : Target "
            "and control qubits must not overlap.");
    }
}

template class GateBase<Prec>;
}  // namespace scaluq::internal
