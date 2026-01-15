#include <scaluq/gate/param_gate_pauli.hpp>
#include <scaluq/operator/apply_pauli.hpp>

namespace scaluq::internal {
template <Precision Prec>
ComplexMatrix ParamPauliRotationGateImpl<Prec>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    StdComplex true_angle = angle * this->_pauli.coef();
    StdComplex half_angle = true_angle / 2.;
    ComplexMatrix mat = this->_pauli.get_matrix_ignoring_coef();
    StdComplex imag_unit(0, 1);
    mat = std::cos(-half_angle) * ComplexMatrix::Identity(mat.rows(), mat.cols()) +
          imag_unit * std::sin(-half_angle) * mat;
    return mat;
}
template <Precision Prec>
std::string ParamPauliRotationGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    auto controls = this->control_qubit_list();
    ss << indent << "Gate Type: ParamPauliRotation\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Pauli Operator: \"" << _pauli.get_pauli_string() << "\"";
    return ss.str();
}
template <Precision Prec>
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Host>& state_vector, double param) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef * static_cast<Float<Prec>>(param),
                         state_vector);
}
template <Precision Prec>
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Host>& states, std::vector<double> params) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef,
                         params_view,
                         states);
}
template <Precision Prec>
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::HostSerialSpace>& state_vector, double param) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef * static_cast<Float<Prec>>(param),
                         state_vector);
}
template <Precision Prec>
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::HostSerialSpace>& states,
    std::vector<double> params) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace> params_view("params_view", params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef,
                         params_view,
                         states);
}
#ifdef SCALUQ_USE_CUDA
template <Precision Prec>
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(
    StateVector<Prec, ExecutionSpace::Default>& state_vector, double param) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef * static_cast<Float<Prec>>(param),
                         state_vector);
    std::cout << "apply end" << std::endl;
}
template <Precision Prec>
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(
    StateVectorBatched<Prec, ExecutionSpace::Default>& states, std::vector<double> params) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    Kokkos::View<Float<Prec>*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> params_view_host(
        params_prec.data(), params_prec.size());
    Kokkos::View<Float<Prec>*, Kokkos::DefaultExecutionSpace> params_view("params_view",
                                                                          params.size());
    Kokkos::deep_copy(params_view, params_view_host);
    apply_pauli_rotation(this->_control_mask,
                         this->_control_value_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef,
                         params_view,
                         states);
}
#endif
template class ParamPauliRotationGateImpl<Prec>;

template <Precision Prec>
std::shared_ptr<const ParamPauliRotationGateImpl<Prec>>
GetParamGateFromJson<ParamPauliRotationGateImpl<Prec>>::get(const Json& j) {
    auto controls = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const ParamPauliRotationGateImpl<Prec>>(
        vector_to_mask(controls),
        vector_to_mask(controls, control_values),
        j.at("pauli").get<PauliOperator<Prec>>(),
        static_cast<Float<Prec>>(j.at("param_coef").get<double>()));
}
template class GetParamGateFromJson<ParamPauliRotationGateImpl<Prec>>;
}  // namespace scaluq::internal
