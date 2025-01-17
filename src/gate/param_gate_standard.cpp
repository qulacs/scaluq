#include <scaluq/gate/param_gate_standard.hpp>

#include "../util/math.hpp"
#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
FLOAT(Fp)
ComplexMatrix<Fp> ParamRXGateImpl<Fp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    Fp half_angle = angle / Fp{2};
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << internal::cos(half_angle), StdComplex<Fp>(0, -internal::sin(half_angle)),
        StdComplex<Fp>(0, -internal::sin(half_angle)), internal::cos(half_angle);
    return mat;
}
FLOAT(Fp)
void ParamRXGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector, Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
}
FLOAT(Fp)
void ParamRXGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states,
                                               std::vector<Fp> params) const {
    this->check_qubit_mask_within_bounds(states);
    rx_gate(this->_target_mask, this->_control_mask, this->_pcoef, params, states);
}
FLOAT(Fp)
std::string ParamRXGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(ParamRXGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> ParamRYGateImpl<Fp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    Fp half_angle = angle / Fp{2};
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << internal::cos(half_angle), -internal::sin(half_angle), internal::sin(half_angle),
        internal::cos(half_angle);
    return mat;
}
FLOAT(Fp)
void ParamRYGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector, Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
}
FLOAT(Fp)
void ParamRYGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states,
                                               std::vector<Fp> params) const {
    this->check_qubit_mask_within_bounds(states);
    ry_gate(this->_target_mask, this->_control_mask, this->_pcoef, params, states);
}
FLOAT(Fp)
std::string ParamRYGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(ParamRYGateImpl)

FLOAT(Fp)
ComplexMatrix<Fp> ParamRZGateImpl<Fp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    Fp half_angle = angle / Fp{2};
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << internal::exp(Complex<Fp>(0, -half_angle)), 0, 0,
        internal::exp(Complex<Fp>(0, half_angle));
    return mat;
}
FLOAT(Fp)
void ParamRZGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector, Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
}
FLOAT(Fp)
void ParamRZGateImpl<Fp>::update_quantum_state(StateVectorBatched<Fp>& states,
                                               std::vector<Fp> params) const {
    this->check_qubit_mask_within_bounds(states);
    rz_gate(this->_target_mask, this->_control_mask, this->_pcoef, params, states);
}
FLOAT(Fp)
std::string ParamRZGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(ParamRZGateImpl)
}  // namespace scaluq::internal
