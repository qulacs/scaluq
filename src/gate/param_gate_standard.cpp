#include <scaluq/gate/param_gate_standard.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec>
ComplexMatrix ParamRXGateImpl<Prec>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::cos(half_angle), StdComplex(0, -std::sin(half_angle)),
        StdComplex(0, -std::sin(half_angle)), std::cos(half_angle);
    return mat;
}
template <Precision Prec>
std::string ParamRXGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template <Precision Prec>
void ParamRXGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Host>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRXGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Host>& states, std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
template <Precision Prec>
void ParamRXGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::HostSerial>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRXGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
    std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
#ifdef SCALUQ_USE_CUDA
template <Precision Prec>
void ParamRXGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Default>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRXGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Default>& states, std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::DefaultExecutionSpace> params_view("params_view",
                                                                          params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
#endif  // SCALUQ_USE_CUDA
template class ParamRXGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix ParamRYGateImpl<Prec>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::cos(half_angle), -std::sin(half_angle), std::sin(half_angle), std::cos(half_angle);
    return mat;
}
template <Precision Prec>
std::string ParamRYGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Host>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Host>& states, std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::HostSerial>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
    std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
#ifdef SCALUQ_USE_CUDA
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Default>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Default>& states, std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::DefaultExecutionSpace> params_view("params_view",
                                                                          params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
#endif  // SCALUQ_USE_CUDA
template class ParamRYGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix ParamRZGateImpl<Prec>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::exp(StdComplex(0, -half_angle)), 0, 0, std::exp(StdComplex(0, half_angle));
    return mat;
}
template <Precision Prec>
std::string ParamRZGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Host>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Host>& states, std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::HostSerial>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::HostSerial>& states,
    std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
#ifdef SCALUQ_USE_CUDA
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Default>& state_vector, double param) const {
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Default>& states, std::vector<double> params) const {
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::DefaultExecutionSpace> params_view("params_view",
                                                                          params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            this->_pcoef,
            params_view,
            states);
}
#endif  // SCALUQ_USE_CUDA
template class ParamRZGateImpl<Prec>;

#define DECLARE_GET_FROM_JSON(Impl)                                                          \
    template <Precision Prec>                                                                \
    std::shared_ptr<const Impl<Prec>> GetParamGateFromJson<Impl<Prec>>::get(const Json& j) { \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                   \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();       \
        return std::make_shared<const Impl<Prec>>(                                           \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),                \
            vector_to_mask(controls),                                                        \
            vector_to_mask(control_values),                                                  \
            static_cast<Float<Prec>>(j.at("param_coef").get<double>()));                     \
    }                                                                                        \
    template struct GetParamGateFromJson<Impl<Prec>>;

DECLARE_GET_FROM_JSON(ParamRXGateImpl)
DECLARE_GET_FROM_JSON(ParamRYGateImpl)
DECLARE_GET_FROM_JSON(ParamRZGateImpl)
}  // namespace scaluq::internal
