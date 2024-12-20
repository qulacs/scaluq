#include <scaluq/gate/gate_standard.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
FLOAT(Fp)
ComplexMatrix<Fp> IGateImpl<Fp>::get_matrix() const {
    return internal::ComplexMatrix<Fp>::Identity(1, 1);
}
FLOAT(Fp)
void IGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    i_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void IGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    i_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string IGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: I\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(IGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> GlobalPhaseGateImpl<Fp>::get_matrix() const {
    return internal::ComplexMatrix<Fp>::Identity(1, 1) * std::exp(std::complex<Fp>(0, _phase));
}
FLOAT(Fp)
void GlobalPhaseGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    global_phase_gate(this->_target_mask, this->_control_mask, _phase, state_vector);
}
FLOAT(Fp)
void GlobalPhaseGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    global_phase_gate(this->_target_mask, this->_control_mask, _phase, states);
}
FLOAT(Fp)
std::string GlobalPhaseGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: GlobalPhase\n";
    ss << indent << "  Phase: " << _phase << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(GlobalPhaseGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> XGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 0, 1, 1, 0;
    return mat;
}
FLOAT(Fp)
void XGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    x_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void XGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    x_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string XGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: X\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(XGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> YGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 0, StdComplex<Fp>(0, -1), StdComplex<Fp>(0, 1), 0;
    return mat;
}
FLOAT(Fp)
void YGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    y_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void YGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    y_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string YGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Y\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(YGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> ZGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, -1;
    return mat;
}
FLOAT(Fp)
void ZGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    z_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void ZGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    z_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string ZGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Z\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(ZGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> HGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 1, 1, -1;
    mat /= (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT(Fp)
void HGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    h_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void HGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    h_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string HGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: H\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(HGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> SGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(0, 1);
    return mat;
}
FLOAT(Fp)
void SGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    s_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void SGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    s_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string SGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: S\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> SdagGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(0, -1);
    return mat;
}
FLOAT(Fp)
void SdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sdag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void SdagGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sdag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string SdagGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Sdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SdagGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> TGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(1, 1) / (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT(Fp)
void TGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    t_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void TGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    t_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string TGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: T\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(TGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> TdagGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, StdComplex<Fp>(1, -1) / (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT(Fp)
void TdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    tdag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void TdagGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    tdag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string TdagGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Tdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(TdagGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> SqrtXGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, 0.5), StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(0.5, -0.5),
        StdComplex<Fp>(0.5, 0.5);
    return mat;
}
FLOAT(Fp)
void SqrtXGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtx_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void SqrtXGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtx_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string SqrtXGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SqrtXGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> SqrtXdagGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(0.5, 0.5), StdComplex<Fp>(0.5, 0.5),
        StdComplex<Fp>(0.5, -0.5);
    return mat;
}
FLOAT(Fp)
void SqrtXdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtxdag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void SqrtXdagGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtxdag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string SqrtXdagGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtXdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SqrtXdagGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> SqrtYGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, 0.5), StdComplex<Fp>(-0.5, -0.5), StdComplex<Fp>(0.5, 0.5),
        StdComplex<Fp>(0.5, 0.5);
    return mat;
}
FLOAT(Fp)
void SqrtYGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrty_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void SqrtYGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrty_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string SqrtYGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SqrtYGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> SqrtYdagGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(0.5, -0.5), StdComplex<Fp>(-0.5, 0.5),
        StdComplex<Fp>(0.5, -0.5);
    return mat;
}
FLOAT(Fp)
void SqrtYdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtydag_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void SqrtYdagGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtydag_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string SqrtYdagGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtYdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SqrtYdagGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> P0GateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, 0;
    return mat;
}
FLOAT(Fp)
void P0GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p0_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void P0GateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    p0_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string P0GateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P0\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(P0GateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> P1GateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 0, 0, 0, 1;
    return mat;
}
FLOAT(Fp)
void P1GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p1_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void P1GateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    p1_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string P1GateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P1\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(P1GateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> RXGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(this->_angle / 2), StdComplex<Fp>(0, -std::sin(this->_angle / 2)),
        StdComplex<Fp>(0, -std::sin(this->_angle / 2)), std::cos(this->_angle / 2);
    return mat;
}
FLOAT(Fp)
void RXGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
}
FLOAT(Fp)
void RXGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    rx_gate(this->_target_mask, this->_control_mask, this->_angle, states);
}
FLOAT(Fp)
std::string RXGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RX\n";
    ss << indent << "  Angle: " << this->_angle << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(RXGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> RYGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(this->_angle / 2), -std::sin(this->_angle / 2), std::sin(this->_angle / 2),
        std::cos(this->_angle / 2);
    return mat;
}
FLOAT(Fp)
void RYGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
}
FLOAT(Fp)
void RYGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    ry_gate(this->_target_mask, this->_control_mask, this->_angle, states);
}
FLOAT(Fp)
std::string RYGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RY\n";
    ss << indent << "  Angle: " << this->_angle << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(RYGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> RZGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::exp(StdComplex<Fp>(0, -0.5 * this->_angle)), 0, 0,
        std::exp(StdComplex<Fp>(0, 0.5 * this->_angle));
    return mat;
}
FLOAT(Fp)
void RZGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
}
FLOAT(Fp)
void RZGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    rz_gate(this->_target_mask, this->_control_mask, this->_angle, states);
}
FLOAT(Fp)
std::string RZGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(RZGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> U1GateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << 1, 0, 0, std::exp(StdComplex<Fp>(0, _lambda));
    return mat;
}
FLOAT(Fp)
void U1GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u1_gate(this->_target_mask, this->_control_mask, _lambda, state_vector);
}
FLOAT(Fp)
void U1GateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u1_gate(this->_target_mask, this->_control_mask, _lambda, states);
}
FLOAT(Fp)
std::string U1GateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U1\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(U1GateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> U2GateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(Kokkos::numbers::pi / 4.),
        -std::exp(StdComplex<Fp>(0, _lambda)) * std::sin((Fp)Kokkos::numbers::pi / 4),
        std::exp(StdComplex<Fp>(0, _phi)) * std::sin((Fp)Kokkos::numbers::pi / 4),
        std::exp(StdComplex<Fp>(0, _phi)) * std::exp(StdComplex<Fp>(0, _lambda)) *
            std::cos((Fp)Kokkos::numbers::pi / 4);
    return mat;
}
FLOAT(Fp)
void U2GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u2_gate(this->_target_mask, this->_control_mask, _phi, _lambda, state_vector);
}
FLOAT(Fp)
void U2GateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u2_gate(this->_target_mask, this->_control_mask, _phi, _lambda, states);
}
FLOAT(Fp)
std::string U2GateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U2\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(U2GateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> U3GateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(_theta / 2.), -std::exp(StdComplex<Fp>(0, _lambda)) * std::sin(_theta / 2),
        std::exp(StdComplex<Fp>(0, _phi)) * std::sin(_theta / 2),
        std::exp(StdComplex<Fp>(0, _phi)) * std::exp(StdComplex<Fp>(0, _lambda)) *
            std::cos(_theta / 2);
    return mat;
}
FLOAT(Fp)
void U3GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u3_gate(this->_target_mask, this->_control_mask, _theta, _phi, _lambda, state_vector);
}
FLOAT(Fp)
void U3GateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u3_gate(this->_target_mask, this->_control_mask, _theta, _phi, _lambda, states);
}
FLOAT(Fp)
std::string U3GateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U3\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(U3GateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> SwapGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat = internal::ComplexMatrix<Fp>::Identity(1 << 2, 1 << 2);
    mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    return mat;
}
FLOAT(Fp)
void SwapGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    swap_gate(this->_target_mask, this->_control_mask, state_vector);
}
FLOAT(Fp)
void SwapGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    swap_gate(this->_target_mask, this->_control_mask, states);
}
FLOAT(Fp)
std::string SwapGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Swap\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SwapGateImpl)
}  // namespace scaluq::internal
