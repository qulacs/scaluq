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

FLOAT(Fp)
ComplexMatrix<Fp> TwoTargetMatrixGateImpl<Fp>::get_matrix() const {
    internal::ComplexMatrix<Fp> mat(4, 4);
    mat << this->_matrix[0][0], this->_matrix[0][1], this->_matrix[0][2], this->_matrix[0][3],
        this->_matrix[1][0], this->_matrix[1][1], this->_matrix[1][2], this->_matrix[1][3],
        this->_matrix[2][0], this->_matrix[2][1], this->_matrix[2][2], this->_matrix[2][3],
        this->_matrix[3][0], this->_matrix[3][1], this->_matrix[3][2], this->_matrix[3][3];
    return mat;
}
FLOAT(Fp)
void TwoTargetMatrixGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    two_target_dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
FLOAT(Fp)
std::string TwoTargetMatrixGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "std::shared_ptr<const GateBase<Fp>> Type: TwoTargetMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(TwoTargetMatrixGateImpl)

FLOAT(Fp)
DenseMatrixGateImpl<Fp>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                             std::uint64_t control_mask,
                                             const ComplexMatrix<Fp>& mat,
                                             bool is_unitary)
    : GateBase<Fp>(target_mask, control_mask),
      _matrix(convert_external_matrix_to_internal_matrix(mat)),
      _is_unitary(is_unitary) {}
FLOAT(Fp)
std::shared_ptr<const GateBase<Fp>> DenseMatrixGateImpl<Fp>::get_inverse() const {
    ComplexMatrix<Fp> mat_eigen = convert_internal_matrix_to_external_matrix(_matrix);
    ComplexMatrix<Fp> inv_eigen;
    if (_is_unitary) {
        inv_eigen = mat_eigen.adjoint();
    } else {
        inv_eigen = mat_eigen.inverse().eval();
    }
    return std::make_shared<const DenseMatrixGateImpl>(
        this->_target_mask, this->_control_mask, inv_eigen, _is_unitary);
}
FLOAT(Fp)
Matrix<Fp> DenseMatrixGateImpl<Fp>::get_matrix_internal() const {
    Matrix<Fp> ret("return matrix", _matrix.extent(0), _matrix.extent(1));
    Kokkos::deep_copy(ret, _matrix);
    return ret;
}
FLOAT(Fp)
ComplexMatrix<Fp> DenseMatrixGateImpl<Fp>::get_matrix() const {
    return convert_internal_matrix_to_external_matrix(_matrix);
}
FLOAT(Fp)
void DenseMatrixGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
FLOAT(Fp)
std::string DenseMatrixGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: DenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(DenseMatrixGateImpl)

FLOAT(Fp)
SparseMatrixGateImpl<Fp>::SparseMatrixGateImpl(std::uint64_t target_mask,
                                               std::uint64_t control_mask,
                                               const SparseComplexMatrix<Fp>& mat)
    : GateBase<Fp>(target_mask, control_mask),
      _matrix(SparseMatrix(mat)),
      num_nnz(mat.nonZeros()) {}
FLOAT(Fp)
std::shared_ptr<const GateBase<Fp>> SparseMatrixGateImpl<Fp>::get_inverse() const {
    Kokkos::View<SparseValue<Fp>*, Kokkos::HostSpace> vec_h("h_view", num_nnz);
    Kokkos::deep_copy(vec_h, _matrix._values);
    // conversion to Eigen matrix (COO format)
    ComplexMatrix<Fp> eigen_matrix(_matrix._row, _matrix._col);
    for (std::size_t i = 0; i < vec_h.extent(0); i++) {
        eigen_matrix(vec_h(i).r, vec_h(i).c) = vec_h(i).val;
    }
    return std::make_shared<const DenseMatrixGateImpl<Fp>>(
        this->_target_mask, this->_control_mask, eigen_matrix.inverse().eval());
}
FLOAT(Fp)
Matrix<Fp> SparseMatrixGateImpl<Fp>::get_matrix_internal() const {
    Matrix<Fp> ret("return matrix", _matrix._row, _matrix._col);
    auto vec = _matrix._values;
    Kokkos::parallel_for(
        vec.size(), KOKKOS_LAMBDA(int i) { ret(vec[i].r, vec[i].c) = vec[i].val; });
    return ret;
}
FLOAT(Fp)
ComplexMatrix<Fp> SparseMatrixGateImpl<Fp>::get_matrix() const {
    return convert_coo_to_external_matrix(_matrix);
}
FLOAT(Fp)
void SparseMatrixGateImpl<Fp>::update_quantum_state(StateVector<Fp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sparse_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
FLOAT(Fp)
std::string SparseMatrixGateImpl<Fp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SparseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_DECLARE_CLASS(SparseMatrixGateImpl)
}  // namespace scaluq::internal
