#include <scaluq/gate/param_gate_standard.hpp>

#include "../prec_space.hpp"
#include "../util/math.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec, ExecutionSpace Space>
ComplexMatrix ParamRXGateImpl<Prec, Space>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::cos(half_angle), StdComplex(0, -std::sin(half_angle)),
        StdComplex(0, -std::sin(half_angle)), std::cos(half_angle);
    return mat;
}
template <Precision Prec, ExecutionSpace Space>
void ParamRXGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector,
                                                        double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec, ExecutionSpace Space>
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
template <Precision Prec, ExecutionSpace Space>
std::string ParamRXGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class ParamRXGateImpl<Prec, Space>;

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix ParamRYGateImpl<Prec, Space>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::cos(half_angle), -std::sin(half_angle), std::sin(half_angle), std::cos(half_angle);
    return mat;
}
template <Precision Prec, ExecutionSpace Space>
void ParamRYGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector,
                                                        double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec, ExecutionSpace Space>
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
template <Precision Prec, ExecutionSpace Space>
std::string ParamRYGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class ParamRYGateImpl<Prec, Space>;

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix ParamRZGateImpl<Prec, Space>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::exp(StdComplex(0, -half_angle)), 0, 0, std::exp(StdComplex(0, half_angle));
    return mat;
}
template <Precision Prec, ExecutionSpace Space>
void ParamRZGateImpl<Prec, Space>::update_quantum_state(StateVector<Prec, Space>& state_vector,
                                                        double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec, ExecutionSpace Space>
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
template <Precision Prec, ExecutionSpace Space>
std::string ParamRZGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class ParamRZGateImpl<Prec, Space>;

#define DECLARE_GET_FROM_JSON(Impl)                                                        \
    template <Precision Prec, ExecutionSpace Space>                                        \
    std::shared_ptr<const Impl<Prec, Space>> GetParamGateFromJson<Impl<Prec, Space>>::get( \
        const Json& j) {                                                                   \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                 \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();     \
        return std::make_shared<const Impl<Prec, Space>>(                                  \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),              \
            vector_to_mask(controls),                                                      \
            vector_to_mask(control_values),                                                \
            static_cast<Float<Prec>>(j.at("param_coef").get<double>()));                   \
    }                                                                                      \
    template class GetParamGateFromJson<Impl<Prec, Space>>;

DECLARE_GET_FROM_JSON(ParamRXGateImpl)
DECLARE_GET_FROM_JSON(ParamRYGateImpl)
DECLARE_GET_FROM_JSON(ParamRZGateImpl)
}  // namespace scaluq::internal
