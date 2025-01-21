#include <scaluq/gate/param_gate_standard.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> ParamRXGateImpl<Fp, Sp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(angle / 2), StdComplex<Fp>(0, -std::sin(angle / 2)),
        StdComplex<Fp>(0, -std::sin(angle / 2)), std::cos(angle / 2);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamRXGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector,
                                                   Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamRXGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                   std::vector<Fp> params) const {
    this->check_qubit_mask_within_bounds(states);
    rx_gate(this->_target_mask, this->_control_mask, this->_pcoef, params, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamRXGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(ParamRXGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> ParamRYGateImpl<Fp, Sp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2);
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamRYGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector,
                                                   Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamRYGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                   std::vector<Fp> params) const {
    this->check_qubit_mask_within_bounds(states);
    ry_gate(this->_target_mask, this->_control_mask, this->_pcoef, params, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamRYGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(ParamRYGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> ParamRZGateImpl<Fp, Sp>::get_matrix(Fp param) const {
    Fp angle = this->_pcoef * param;
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << std::exp(StdComplex<Fp>(0, -angle / 2)), 0, 0, std::exp(StdComplex<Fp>(0, angle / 2));
    return mat;
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamRZGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector,
                                                   Fp param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask, this->_control_mask, this->_pcoef * param, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void ParamRZGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                   std::vector<Fp> params) const {
    this->check_qubit_mask_within_bounds(states);
    rz_gate(this->_target_mask, this->_control_mask, this->_pcoef, params, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamRZGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(ParamRZGateImpl)
}  // namespace scaluq::internal
