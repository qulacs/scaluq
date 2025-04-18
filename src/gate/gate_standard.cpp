#include <scaluq/gate/gate_standard.hpp>

#include "../util/math.hpp"
#include "../prec_space.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <>
ComplexMatrix IGateImpl<Prec, Space>::get_matrix() const {
    return ComplexMatrix::Identity(1, 1);
}
template <>
void IGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    i_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void IGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    i_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string IGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: I\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class IGateImpl<Prec, Space>;

template <>
ComplexMatrix GlobalPhaseGateImpl<Prec, Space>::get_matrix() const {
    return ComplexMatrix::Identity(1, 1) * std::exp(StdComplex(0, static_cast<double>(_phase)));
}
template <>
void GlobalPhaseGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    global_phase_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _phase, state_vector);
}
template <>
void GlobalPhaseGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    global_phase_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _phase, states);
}
template <>
std::string GlobalPhaseGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: GlobalPhase\n";
    ss << indent << "  Phase: " << static_cast<double>(_phase) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class GlobalPhaseGateImpl<Prec, Space>;

template <>
ComplexMatrix XGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 0, 1, 1, 0;
    return mat;
}
template <>
void XGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    x_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void XGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    x_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string XGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: X\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class XGateImpl<Prec, Space>;

template <>
ComplexMatrix YGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 0, StdComplex(0, -1), StdComplex(0, 1), 0;
    return mat;
}
template <>
void YGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    y_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void YGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    y_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string YGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Y\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class YGateImpl<Prec, Space>;

template <>
ComplexMatrix ZGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, -1;
    return mat;
}
template <>
void ZGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    z_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void ZGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    z_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string ZGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Z\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class ZGateImpl<Prec, Space>;

template <>
ComplexMatrix HGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 1, 1, -1;
    mat /= Kokkos::numbers::sqrt2;
    return mat;
}
template <>
void HGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    h_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void HGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    h_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string HGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: H\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class HGateImpl<Prec, Space>;

template <>
ComplexMatrix SGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(0, 1);
    return mat;
}
template <>
void SGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    s_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void SGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    s_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string SGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: S\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SGateImpl<Prec, Space>;

template <>
ComplexMatrix SdagGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(0, -1);
    return mat;
}
template <>
void SdagGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sdag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void SdagGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sdag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string SdagGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Sdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SdagGateImpl<Prec, Space>;

template <>
ComplexMatrix TGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(1, 1) / Kokkos::numbers::sqrt2;
    return mat;
}
template <>
void TGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    t_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void TGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    t_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string TGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: T\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class TGateImpl<Prec, Space>;

template <>
ComplexMatrix TdagGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(1, -1) / Kokkos::numbers::sqrt2;
    return mat;
}
template <>
void TdagGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    tdag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void TdagGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    tdag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string TdagGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Tdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class TdagGateImpl<Prec, Space>;

template <>
ComplexMatrix SqrtXGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, .5), StdComplex(.5, -.5), StdComplex(.5, -.5), StdComplex(.5, .5);
    return mat;
}
template <>
void SqrtXGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtx_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void SqrtXGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtx_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string SqrtXGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SqrtXGateImpl<Prec, Space>;

template <>
ComplexMatrix SqrtXdagGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, -.5), StdComplex(.5, .5), StdComplex(.5, .5), StdComplex(.5, -.5);
    return mat;
}
template <>
void SqrtXdagGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtxdag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void SqrtXdagGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtxdag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string SqrtXdagGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtXdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SqrtXdagGateImpl<Prec, Space>;

template <>
ComplexMatrix SqrtYGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, .5), StdComplex(-.5, -.5), StdComplex(.5, .5), StdComplex(.5, .5);
    return mat;
}
template <>
void SqrtYGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrty_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void SqrtYGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrty_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string SqrtYGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SqrtYGateImpl<Prec, Space>;

template <>
ComplexMatrix SqrtYdagGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, -.5), StdComplex(.5, -.5), StdComplex(-.5, .5), StdComplex(.5, -.5);
    return mat;
}
template <>
void SqrtYdagGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtydag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void SqrtYdagGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sqrtydag_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string SqrtYdagGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtYdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SqrtYdagGateImpl<Prec, Space>;

template <>
ComplexMatrix P0GateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, 0;
    return mat;
}
template <>
void P0GateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p0_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void P0GateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    p0_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string P0GateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P0\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class P0GateImpl<Prec, Space>;

template <>
ComplexMatrix P1GateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 0, 0, 0, 1;
    return mat;
}
template <>
void P1GateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p1_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void P1GateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    p1_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string P1GateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P1\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class P1GateImpl<Prec, Space>;

template <>
ComplexMatrix RXGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    double half_angle = static_cast<double>(this->_angle) / 2;
    mat << std::cos(half_angle), StdComplex(0, -std::sin(half_angle)),
        StdComplex(0, -std::sin(half_angle)), std::cos(half_angle);
    return mat;
}
template <>
void RXGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_angle,
            state_vector);
}
template <>
void RXGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    rx_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, this->_angle, states);
}
template <>
std::string RXGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RX\n";
    ss << indent << "  Angle: " << static_cast<double>(this->_angle) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class RXGateImpl<Prec, Space>;

template <>
ComplexMatrix RYGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    double half_angle = static_cast<double>(this->_angle) / 2;
    mat << std::cos(half_angle), -std::sin(half_angle), std::sin(half_angle), std::cos(half_angle);
    return mat;
}
template <>
void RYGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_angle,
            state_vector);
}
template <>
void RYGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    ry_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, this->_angle, states);
}
template <>
std::string RYGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RY\n";
    ss << indent << "  Angle: " << static_cast<double>(this->_angle) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class RYGateImpl<Prec, Space>;

template <>
ComplexMatrix RZGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    double half_angle = static_cast<double>(this->_angle) / 2;
    mat << std::exp(StdComplex(0, -half_angle)), 0, 0, std::exp(StdComplex(0, half_angle));
    return mat;
}
template <>
void RZGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_angle,
            state_vector);
}
template <>
void RZGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    rz_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, this->_angle, states);
}
template <>
std::string RZGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RZ\n";
    ss << indent << "  Angle: " << static_cast<double>(this->_angle) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class RZGateImpl<Prec, Space>;

template <>
ComplexMatrix U1GateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, std::exp(StdComplex(0, static_cast<double>(_lambda)));
    return mat;
}
template <>
void U1GateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u1_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _lambda, state_vector);
}
template <>
void U1GateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u1_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, _lambda, states);
}
template <>
std::string U1GateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U1\n";
    ss << indent << "  Lambda: " << static_cast<double>(this->_lambda) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class U1GateImpl<Prec, Space>;

template <>
ComplexMatrix U2GateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << std::cos(Kokkos::numbers::pi / 4.),
        -std::exp(StdComplex(0, static_cast<double>(_lambda))) * std::sin(Kokkos::numbers::pi / 4),
        std::exp(StdComplex(0, static_cast<double>(_phi))) * std::sin(Kokkos::numbers::pi / 4),
        std::exp(StdComplex(0, static_cast<double>(_phi))) *
            std::exp(StdComplex(0, static_cast<double>(_lambda))) *
            std::cos(Kokkos::numbers::pi / 4);
    return mat;
}
template <>
void U2GateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u2_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            _phi,
            _lambda,
            state_vector);
}
template <>
void U2GateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u2_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _phi, _lambda, states);
}
template <>
std::string U2GateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "  Phi: " << static_cast<double>(this->_phi) << "\n";
    ss << indent << "  Lambda: " << static_cast<double>(this->_lambda) << "\n";
    ss << indent << "Gate Type: U2\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class U2GateImpl<Prec, Space>;

template <>
ComplexMatrix U3GateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << std::cos(static_cast<double>(_theta) / 2),
        -std::exp(StdComplex(0, static_cast<double>(_lambda))) *
            std::sin(static_cast<double>(_theta) / 2),
        std::exp(StdComplex(0, static_cast<double>(_phi))) *
            std::sin(static_cast<double>(_theta) / 2),
        std::exp(StdComplex(0, static_cast<double>(_phi))) *
            std::exp(StdComplex(0, static_cast<double>(_lambda))) *
            std::cos(static_cast<double>(_theta) / 2);
    return mat;
}
template <>
void U3GateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u3_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            _theta,
            _phi,
            _lambda,
            state_vector);
}
template <>
void U3GateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    u3_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            _theta,
            _phi,
            _lambda,
            states);
}
template <>
std::string U3GateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U3\n";
    ss << indent << "  Theta: " << static_cast<double>(this->_theta) << "\n";
    ss << indent << "  Phi: " << static_cast<double>(this->_phi) << "\n";
    ss << indent << "  Lambda: " << static_cast<double>(this->_lambda) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class U3GateImpl<Prec, Space>;

template <>
ComplexMatrix SwapGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix mat = ComplexMatrix::Identity(1 << 2, 1 << 2);
    mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    return mat;
}
template <>
void SwapGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    swap_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);
}
template <>
void SwapGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    swap_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, states);
}
template <>
std::string SwapGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Swap\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SwapGateImpl<Prec, Space>;
}  // namespace scaluq::internal
