#include <scaluq/gate/gate_matrix.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
FLOAT_AND_SPACE(Fp, Sp)
DenseMatrixGateImpl<Fp, Sp>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                                 std::uint64_t control_mask,
                                                 const ComplexMatrix<Fp>& mat,
                                                 bool is_unitary)
    : GateBase<Fp, Sp>(target_mask, control_mask),
      _matrix(convert_external_matrix_to_internal_matrix(mat)),
      _is_unitary(is_unitary) {}
FLOAT_AND_SPACE(Fp, Sp)
std::shared_ptr<const GateBase<Fp, Sp>> DenseMatrixGateImpl<Fp, Sp>::get_inverse() const {
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
FLOAT_AND_SPACE(Fp, Sp)
Matrix<Fp> DenseMatrixGateImpl<Fp, Sp>::get_matrix_internal() const {
    Matrix<Fp> ret("return matrix", _matrix.extent(0), _matrix.extent(1));
    Kokkos::deep_copy(ret, _matrix);
    return ret;
}
FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> DenseMatrixGateImpl<Fp, Sp>::get_matrix() const {
    return convert_internal_matrix_to_external_matrix(_matrix);
}
FLOAT_AND_SPACE(Fp, Sp)
void DenseMatrixGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void DenseMatrixGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string DenseMatrixGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: DenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(DenseMatrixGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
SparseMatrixGateImpl<Fp, Sp>::SparseMatrixGateImpl(std::uint64_t target_mask,
                                                   std::uint64_t control_mask,
                                                   const SparseComplexMatrix<Fp>& mat)
    : GateBase<Fp, Sp>(target_mask, control_mask),
      _matrix(SparseMatrix(mat)),
      num_nnz(mat.nonZeros()) {}
FLOAT_AND_SPACE(Fp, Sp)
std::shared_ptr<const GateBase<Fp, Sp>> SparseMatrixGateImpl<Fp, Sp>::get_inverse() const {
    Kokkos::View<SparseValue<Fp>*, Kokkos::HostSpace> vec_h("h_view", num_nnz);
    Kokkos::deep_copy(vec_h, _matrix._values);
    // conversion to Eigen matrix (COO format)
    ComplexMatrix<Fp> eigen_matrix = ComplexMatrix<Fp>::Zero(_matrix._row, _matrix._col);
    for (std::size_t i = 0; i < vec_h.extent(0); i++) {
        eigen_matrix(vec_h(i).r, vec_h(i).c) = vec_h(i).val;
    }
    return std::make_shared<const DenseMatrixGateImpl<Fp, Sp>>(
        this->_target_mask, this->_control_mask, eigen_matrix.inverse().eval());
}
FLOAT_AND_SPACE(Fp, Sp)
Matrix<Fp> SparseMatrixGateImpl<Fp, Sp>::get_matrix_internal() const {
    Matrix<Fp> ret("return matrix", _matrix._row, _matrix._col);
    auto vec = _matrix._values;
    Kokkos::parallel_for(
        vec.size(), KOKKOS_LAMBDA(int i) { ret(vec[i].r, vec[i].c) = vec[i].val; });
    return ret;
}
FLOAT_AND_SPACE(Fp, Sp)
ComplexMatrix<Fp> SparseMatrixGateImpl<Fp, Sp>::get_matrix() const {
    return convert_coo_to_external_matrix(_matrix);
}
FLOAT_AND_SPACE(Fp, Sp)
void SparseMatrixGateImpl<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sparse_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
FLOAT_AND_SPACE(Fp, Sp)
void SparseMatrixGateImpl<Fp, Sp>::update_quantum_state(StateVectorBatched<Fp, Sp>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sparse_matrix_gate(this->_target_mask, this->_control_mask, _matrix, states);
}
FLOAT_AND_SPACE(Fp, Sp)
std::string SparseMatrixGateImpl<Fp, Sp>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SparseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
FLOAT_AND_SPACE_DECLARE_CLASS(SparseMatrixGateImpl)
}  // namespace scaluq::internal
