#include <scaluq/gate/gate_standard.hpp>

#include "../util/math.hpp"
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
std::string IGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: I\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(IGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> GlobalPhaseGateImpl<Fp>::get_matrix() const {
    return internal::ComplexMatrix<Fp>::Identity(1, 1) * internal::exp(Complex<Fp>(Fp{0}, _phase));
}
FLOAT(Fp)
void GlobalPhaseGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    global_phase_gate(this->_target_mask, this->_control_mask, _phase, state_vector);
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
    mat << Fp{0}, Fp{1}, Fp{1}, Fp{0};
    return mat;
}
FLOAT(Fp)
void XGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    x_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{0}, StdComplex<Fp>(Fp{0}, -Fp{1}), StdComplex<Fp>(Fp{0}, Fp{1}), Fp{0};
    return mat;
}
FLOAT(Fp)
void YGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    y_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, -Fp{1};
    return mat;
}
FLOAT(Fp)
void ZGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    z_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{1}, Fp{1}, Fp{1}, -Fp{1};
    mat /= (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT(Fp)
void HGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    h_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, StdComplex<Fp>(0, 1);
    return mat;
}
FLOAT(Fp)
void SGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    s_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, StdComplex<Fp>(0, -1);
    return mat;
}
FLOAT(Fp)
void SdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sdag_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, StdComplex<Fp>(1, 1) / (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT(Fp)
void TGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    t_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, StdComplex<Fp>(1, -1) / (Fp)Kokkos::numbers::sqrt2;
    return mat;
}
FLOAT(Fp)
void TdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    tdag_gate(this->_target_mask, this->_control_mask, state_vector);
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
    Fp half = static_cast<Fp>(0.5);
    mat << StdComplex<Fp>(half, half), StdComplex<Fp>(half, -half), StdComplex<Fp>(half, -half),
        StdComplex<Fp>(half, half);
    return mat;
}
FLOAT(Fp)
void SqrtXGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtx_gate(this->_target_mask, this->_control_mask, state_vector);
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
    Fp half = static_cast<Fp>(0.5);
    mat << StdComplex<Fp>(half, -half), StdComplex<Fp>(half, half), StdComplex<Fp>(half, half),
        StdComplex<Fp>(half, -half);
    return mat;
}
FLOAT(Fp)
void SqrtXdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtxdag_gate(this->_target_mask, this->_control_mask, state_vector);
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
    Fp half = static_cast<Fp>(0.5);
    mat << StdComplex<Fp>(half, half), StdComplex<Fp>(-half, -half), StdComplex<Fp>(half, half),
        StdComplex<Fp>(half, half);
    return mat;
}
FLOAT(Fp)
void SqrtYGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrty_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{0}, StdComplex<Fp>(0, -1), StdComplex<Fp>(0, 1), Fp{0};
    return mat;
}
FLOAT(Fp)
void SqrtYdagGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sqrtydag_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, Fp{0};
    return mat;
}
FLOAT(Fp)
void P0GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p0_gate(this->_target_mask, this->_control_mask, state_vector);
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
    mat << Fp{0}, Fp{0}, Fp{0}, Fp{1};
    return mat;
}
FLOAT(Fp)
void P1GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    p1_gate(this->_target_mask, this->_control_mask, state_vector);
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
    Fp half_angle = this->_angle / Fp{2};
    mat << internal::cos(half_angle), Complex<Fp>(Fp{0}, -internal::sin(half_angle)),
        Complex<Fp>(Fp{0}, internal::sin(half_angle)), internal::cos(half_angle);
    return mat;
}
FLOAT(Fp)
void RXGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
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
    Fp half_angle = this->_angle / Fp{2};
    mat << internal::cos(half_angle), -internal::sin(half_angle), internal::sin(half_angle),
        internal::cos(half_angle);
    return mat;
}
FLOAT(Fp)
void RYGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
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
    Fp half = static_cast<Fp>(0.5);
    mat << internal::exp(Complex<Fp>(Fp{0}, -half * this->_angle)), Fp{0}, Fp{0},
        internal::exp(Complex<Fp>(Fp{0}, half * this->_angle));
    return mat;
}
FLOAT(Fp)
void RZGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, internal::exp(Complex<Fp>(Fp{0}, _lambda));
    return mat;
}
FLOAT(Fp)
void U1GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u1_gate(this->_target_mask, this->_control_mask, _lambda, state_vector);
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
    mat << internal::cos(static_cast<Fp>(Kokkos::numbers::pi / 4.)),
        -internal::exp(Complex<Fp>(Fp{0}, _lambda)) *
            internal::sin((Fp)Kokkos::numbers::pi / Fp{4}),
        internal::exp(Complex<Fp>(Fp{0}, _phi)) * internal::sin((Fp)Kokkos::numbers::pi / Fp{4}),
        internal::exp(Complex<Fp>(Fp{0}, _phi)) * internal::exp(Complex<Fp>(Fp{0}, _lambda)) *
            internal::cos((Fp)Kokkos::numbers::pi / Fp{4});
    return mat;
}
FLOAT(Fp)
void U2GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u2_gate(this->_target_mask, this->_control_mask, _phi, _lambda, state_vector);
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
    mat << internal::cos(_theta / Fp{2}),
        -internal::exp(Complex<Fp>(Fp{0}, _lambda)) * internal::sin(_theta / Fp{2}),
        internal::exp(Complex<Fp>(Fp{0}, _phi)) * internal::sin(_theta / Fp{2}),
        internal::exp(Complex<Fp>(Fp{0}, _phi)) * internal::exp(Complex<Fp>(Fp{0}, _lambda)) *
            internal::cos(_theta / Fp{2});
    return mat;
}
FLOAT(Fp)
void U3GateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    u3_gate(this->_target_mask, this->_control_mask, _theta, _phi, _lambda, state_vector);
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
    mat << Fp{1}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{0}, Fp{1}, Fp{0}, Fp{0}, Fp{1}, Fp{0}, Fp{0},
        Fp{0}, Fp{0}, Fp{0}, Fp{1};
    return mat;
}
FLOAT(Fp)
void SwapGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    swap_gate(this->_target_mask, this->_control_mask, state_vector);
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
