#include <scaluq/gate/param_gate_pauli.hpp>

#include "../operator/apply_pauli.hpp"
#include "../util/math.hpp"
#include "../util/template.hpp"

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
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector,
                                                            double param) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    apply_pauli_rotation(this->_control_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef * Float<Prec>{param},
                         state_vector);
}
template <Precision Prec>
void ParamPauliRotationGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states,
                                                            std::vector<double> params) const {
    auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(params, params_prec.begin(), [](double p) { return Float<Prec>{p}; });
    apply_pauli_rotation(this->_control_mask,
                         bit_flip_mask,
                         phase_flip_mask,
                         Complex<Prec>(_pauli.coef()),
                         this->_pcoef,
                         params_prec,
                         states);
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
SCALUQ_DECLARE_CLASS_FOR_PRECISION(ParamPauliRotationGateImpl)
}  // namespace scaluq::internal
