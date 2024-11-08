#include <scaluq/gate/gate_matrix.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
FLOAT(Fp)
ComplexMatrix<Fp> OneTargetMatrixGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(2, 2);
    mat << this->_matrix[0][0], this->_matrix[0][1], this->_matrix[1][0], this->_matrix[1][1];
    return mat;
}
FLOAT(Fp)
void OneTargetMatrixGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    one_target_dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
FLOAT(Fp)
std::string OneTargetMatrixGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "std::shared_ptr<const GateBase<Fp>> Type: OneTargetMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(OneTargetMatrixGateImpl)
}  // namespace scaluq::internal
