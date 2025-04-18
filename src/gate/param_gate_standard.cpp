#include <scaluq/gate/param_gate_standard.hpp>

#include "../util/math.hpp"
#include "../prec_space.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <>
ComplexMatrix ParamRXGateImpl<Prec, Space>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::cos(half_angle), StdComplex(0, -std::sin(half_angle)),
        StdComplex(0, -std::sin(half_angle)), std::cos(half_angle);
    return mat;
}
template <>
void ParamRXGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector,
                                                        double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <>
void ParamRXGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states,
                                                        std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_prec,
            states);
}
template <>
std::string ParamRXGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class ParamRXGateImpl<Prec, Space>;

template <>
ComplexMatrix ParamRYGateImpl<Prec, Space>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::cos(half_angle), -std::sin(half_angle), std::sin(half_angle), std::cos(half_angle);
    return mat;
}
template <>
void ParamRYGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector,
                                                        double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <>
void ParamRYGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states,
                                                        std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_prec,
            states);
}
template <>
std::string ParamRYGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class ParamRYGateImpl<Prec, Space>;

template <>
ComplexMatrix ParamRZGateImpl<Prec, Space>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::exp(StdComplex(0, -half_angle)), 0, 0, std::exp(StdComplex(0, half_angle));
    return mat;
}
template <>
void ParamRZGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector,
                                                        double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <>
void ParamRZGateImpl<Prec, Space>::update_quantum_state(StateVectorBatched<Prec, Space>& states,
                                                        std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_prec,
            states);
}
template <>
std::string ParamRZGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class ParamRZGateImpl<Prec, Space>;
}  // namespace scaluq::internal
