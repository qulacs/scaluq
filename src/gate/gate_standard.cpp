#include <scaluq/gate/gate_standard.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> IGateImpl<Fp, Sp>::get_matrix() const {
    return internal::ComplexMatrix<Fp>::Identity(1, 1);
}
FLOAT_AND_SPACE(Fp, Sp)
void IGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    i_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void IGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    i_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string IGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: I\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(IGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> GlobalPhaseGateImpl<Fp, Sp>::get_matrix() const {
    return internal::ComplexMatrix<Fp>::Identity(1, 1) * std::exp(StdComplex<Fp>(0, _phase));
}
FLOAT_AND_SPACE(Fp, Sp)
void GlobalPhaseGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    global_phase_gate(this->_target_mask, this->_control_mask, _phase, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void GlobalPhaseGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    global_phase_gate(this->_target_mask, this->_control_mask, _phase, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string GlobalPhaseGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: GlobalPhase\n";
    ss << indent << "  Phase: " << _phase << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(GlobalPhaseGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> XGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 0, 1, 1, 0;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void XGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    x_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void XGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    x_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string XGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: X\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(XGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> YGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 0, StdComplex<Fp>(0, -1), StdComplex<Fp>(0, 1), 0;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void YGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    y_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void YGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    y_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string YGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Y\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(YGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> ZGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, -1;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void ZGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    z_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void ZGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    z_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string ZGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Z\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(ZGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> HGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 1, 1, -1;
    mat /= (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void HGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    h_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void HGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    h_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string HGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: H\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(HGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(0, 1);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void SGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    s_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    s_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: S\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SdagGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(0, -1);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void SdagGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sdag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SdagGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sdag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SdagGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Sdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SdagGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> TGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(1, 1) / (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void TGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    t_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void TGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    t_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string TGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: T\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(TGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> TdagGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(1, -1) / (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void TdagGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    tdag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void TdagGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    tdag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string TdagGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Tdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(TdagGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SqrtXGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, 0.5), StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(0.5, -0.5),
        StdComplex<Fp>(0.5, 0.5);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtXGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtx_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtXGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtx_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SqrtXGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SqrtXGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SqrtXdagGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(0.5, 0.5), StdComplex<Fp>(0.5, 0.5),
        StdComplex<Fp>(0.5, -0.5);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtXdagGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtxdag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtXdagGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtxdag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SqrtXdagGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtXdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SqrtXdagGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SqrtYGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, 0.5), StdComplex<Fp>(-0.5, -0.5), StdComplex<Fp>(0.5, 0.5),
        StdComplex<Fp>(0.5, 0.5);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtYGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrty_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtYGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrty_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SqrtYGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SqrtYGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SqrtYdagGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(-0.5, 0.5),
        StdComplex<Fp>(0.5, -0.5);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtYdagGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtydag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SqrtYdagGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtydag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SqrtYdagGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtYdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SqrtYdagGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> P0GateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, 0;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void P0GateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p0_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void P0GateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    p0_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string P0GateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P0\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(P0GateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> P1GateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 0, 0, 0, 1;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void P1GateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p1_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void P1GateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    p1_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string P1GateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P1\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(P1GateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> RXGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(this->_angle / 2), StdComplex<Fp>(0, -std::sin(this->_angle / 2)),
        StdComplex<Fp>(0, -std::sin(this->_angle / 2)), std::cos(this->_angle / 2);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void RXGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void RXGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    rx_gate(this->_target_mask, this->_control_mask, this->_angle, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string RXGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RX\n";
    ss << indent << "  Angle: " << this->_angle << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(RXGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> RYGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(this->_angle / 2), -std::sin(this->_angle / 2), std::sin(this->_angle / 2),
        std::cos(this->_angle / 2);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void RYGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void RYGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    ry_gate(this->_target_mask, this->_control_mask, this->_angle, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string RYGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RY\n";
    ss << indent << "  Angle: " << this->_angle << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(RYGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> RZGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::exp(StdComplex<Fp>(0, -0.5 * this->_angle)), 0, 0,
        std::exp(StdComplex<Fp>(0, 0.5 * this->_angle));
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void RZGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void RZGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    rz_gate(this->_target_mask, this->_control_mask, this->_angle, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string RZGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(RZGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> U1GateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, std::exp(StdComplex<Fp>(0, _lambda));
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void U1GateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u1_gate(this->_target_mask, this->_control_mask, _lambda, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void U1GateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u1_gate(this->_target_mask, this->_control_mask, _lambda, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string U1GateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U1\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(U1GateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> U2GateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(Kokkos::numbers::pi / 4.),
        -std::exp(StdComplex<Fp>(0, _lambda)) * std::sin((Fp)Kokkos::numbers::pi / 4),
        std::exp(StdComplex<Fp>(0, _phi)) * std::sin((Fp)Kokkos::numbers::pi / 4),
        std::exp(StdComplex<Fp>(0, _phi)) * std::exp(StdComplex<Fp>(0, _lambda)) *
            std::cos((Fp)Kokkos::numbers::pi / 4);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void U2GateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u2_gate(this->_target_mask, this->_control_mask, _phi, _lambda, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void U2GateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u2_gate(this->_target_mask, this->_control_mask, _phi, _lambda, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string U2GateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U2\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(U2GateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> U3GateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(_theta / 2.), -std::exp(StdComplex<Fp>(0, _lambda)) * std::sin(_theta / 2),
        std::exp(StdComplex<Fp>(0, _phi)) * std::sin(_theta / 2),
        std::exp(StdComplex<Fp>(0, _phi)) * std::exp(StdComplex<Fp>(0, _lambda)) *
            std::cos(_theta / 2);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void U3GateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u3_gate(this->_target_mask, this->_control_mask, _theta, _phi, _lambda, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void U3GateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u3_gate(this->_target_mask, this->_control_mask, _theta, _phi, _lambda, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string U3GateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U3\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(U3GateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SwapGateImpl<Fp, Sp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat = internal::ComplexMatrix<Fp>::Identity(1 << 2, 1 << 2);
    mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void SwapGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    swap_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SwapGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    swap_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SwapGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Swap\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SwapGateImpl)
}  // namespace scaluq::internal
