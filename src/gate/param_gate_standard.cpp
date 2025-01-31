#include <scaluq/gate/param_gate_standard.hpp>

#include "../util/math.hpp"
#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
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
void ParamRXGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector,
                                                 double param) const {
=======
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
>>>>>>> set-space
    this->check_qubit_mask_within_bounds(state_vector);
    rx_gate(this->_target_mask,
            this->_control_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
<<<<<<< HEAD
template <Precision Prec>
void ParamRXGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states,
                                                 std::vector<double> params) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ParamRXGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                   std::vector<Fp> params) const {
>>>>>>> set-space
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    rx_gate(this->_target_mask, this->_control_mask, this->_pcoef, params_prec, states);
}
<<<<<<< HEAD
template <Precision Prec>
std::string ParamRXGateImpl<Prec>::to_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamRXGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
>>>>>>> set-space
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(ParamRXGateImpl)

template <Precision Prec>
ComplexMatrix ParamRYGateImpl<Prec>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::cos(half_angle), -std::sin(half_angle), std::sin(half_angle), std::cos(half_angle);
    return mat;
}
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector,
                                                 double param) const {
=======
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
>>>>>>> set-space
    this->check_qubit_mask_within_bounds(state_vector);
    ry_gate(this->_target_mask,
            this->_control_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
<<<<<<< HEAD
template <Precision Prec>
void ParamRYGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states,
                                                 std::vector<double> params) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ParamRYGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                   std::vector<Fp> params) const {
>>>>>>> set-space
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    ry_gate(this->_target_mask, this->_control_mask, this->_pcoef, params_prec, states);
}
<<<<<<< HEAD
template <Precision Prec>
std::string ParamRYGateImpl<Prec>::to_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamRYGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
>>>>>>> set-space
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(ParamRYGateImpl)

template <Precision Prec>
ComplexMatrix ParamRZGateImpl<Prec>::get_matrix(double param) const {
    double angle = static_cast<double>(this->_pcoef) * param;
    double half_angle = angle / 2;
    ComplexMatrix mat(2, 2);
    mat << std::exp(StdComplex(0, -half_angle)), 0, 0, std::exp(StdComplex(0, half_angle));
    return mat;
}
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector,
                                                 double param) const {
=======
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
>>>>>>> set-space
    this->check_qubit_mask_within_bounds(state_vector);
    rz_gate(this->_target_mask,
            this->_control_mask,
            this->_pcoef * static_cast<Float<Prec>>(param),
            state_vector);
}
<<<<<<< HEAD
template <Precision Prec>
void ParamRZGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states,
                                                 std::vector<double> params) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
void ParamRZGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states,
                                                   std::vector<Fp> params) const {
>>>>>>> set-space
    this->check_qubit_mask_within_bounds(states);
    std::vector<Float<Prec>> params_prec(params.size());
    std::ranges::transform(
        params, params_prec.begin(), [](double p) { return static_cast<Float<Prec>>(p); });
    rz_gate(this->_target_mask, this->_control_mask, this->_pcoef, params_prec, states);
}
<<<<<<< HEAD
template <Precision Prec>
std::string ParamRZGateImpl<Prec>::to_string(const std::string& indent) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string ParamRZGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
>>>>>>> set-space
    std::ostringstream ss;
    ss << indent << "Gate Type: ParamRZ\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(ParamRZGateImpl)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(ParamRZGateImpl)
>>>>>>> set-space
}  // namespace scaluq::internal
