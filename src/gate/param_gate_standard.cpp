#include <scaluq/gate/param_gate_standard.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
FLOAT(Fp)
ComplexMatrix<Fp> ParamRXGateImpl<Fp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(angle / 2), StdComplex<Fp>(0, -std::sin(angle / 2)),
        StdComplex<Fp>(0, -std::sin(angle / 2)), std::cos(angle / 2);
    return mat;
}
FLOAT(Fp)
void ParamRXGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector, Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
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
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2);
    return mat;
}
FLOAT(Fp)
void ParamRYGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector, Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
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
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::exp(StdComplex<Fp>(0, -angle / 2)), 0, 0, std::exp(StdComplex<Fp>(0, angle / 2));
    return mat;
}
FLOAT(Fp)
void ParamRZGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector, Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
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
