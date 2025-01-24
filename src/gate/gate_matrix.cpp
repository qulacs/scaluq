#include <scaluq/gate/gate_matrix.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec>
DenseMatrixGateImpl<Prec>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                               std::uint64_t control_mask,
                                               const ComplexMatrix& mat,
                                               bool is_unitary)
    : GateBase<Prec>(target_mask, control_mask),
      _matrix(convert_external_matrix_to_internal_matrix(mat)),
      _is_unitary(is_unitary) {}
template <Precision Prec>
std::shared_ptr<const GateBase<Prec>> DenseMatrixGateImpl<Prec>::get_inverse() const {
    ComplexMatrix mat_eigen = convert_internal_matrix_to_external_matrix(_matrix);
    ComplexMatrix inv_eigen;
    if (_is_unitary) {
        inv_eigen = mat_eigen.adjoint();
    } else {
        // inv_eigen = mat_eigen.inverse().eval();
        throw std::runtime_error("inverse of non-unitary matrix gate is currently not available.");
    }
    return std::make_shared<const DenseMatrixGateImpl>(
        this->_target_mask, this->_control_mask, inv_eigen, _is_unitary);
}
template <Precision Prec>
Matrix<Prec> DenseMatrixGateImpl<Prec>::get_matrix_internal() const {
    Matrix<Prec> ret("return matrix", _matrix.extent(0), _matrix.extent(1));
    Kokkos::deep_copy(ret, _matrix);
    return ret;
}
template <Precision Prec>
ComplexMatrix DenseMatrixGateImpl<Prec>::get_matrix() const {
    return convert_internal_matrix_to_external_matrix(_matrix);
}
template <Precision Prec>
void DenseMatrixGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
template <Precision Prec>
void DenseMatrixGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states) const {
    this->check_qubit_mask_within_bounds(states);
    dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, states);
}
template <Precision Prec>
std::string DenseMatrixGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: DenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
SCALUQ_DECLARE_CLASS_FOR_PRECISION(DenseMatrixGateImpl)

template <Precision Prec>
SparseMatrixGateImpl<Prec>::SparseMatrixGateImpl(std::uint64_t target_mask,
                                                 std::uint64_t control_mask,
                                                 const SparseComplexMatrix& mat)
    : GateBase<Prec>(target_mask, control_mask),
      _matrix(SparseMatrix(mat)),
      num_nnz(mat.nonZeros()) {}
template <Precision Prec>
std::shared_ptr<const GateBase<Prec>> SparseMatrixGateImpl<Prec>::get_inverse() const {
    throw std::runtime_error("inverse of sparse matrix gate is currently not available.");
}
template <Precision Prec>
Matrix<Prec> SparseMatrixGateImpl<Prec>::get_matrix_internal() const {
    Matrix<Prec> ret("return matrix", _matrix._row, _matrix._col);
    auto vec = _matrix._values;
    Kokkos::parallel_for(
        vec.size(), KOKKOS_LAMBDA(int i) { ret(vec[i].r, vec[i].c) = vec[i].val; });
    return ret;
}
template <Precision Prec>
ComplexMatrix SparseMatrixGateImpl<Prec>::get_matrix() const {
    return convert_coo_to_external_matrix(_matrix);
}
template <Precision Prec>
void SparseMatrixGateImpl<Prec>::update_quantum_state(StateVector<Prec>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sparse_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
}
template <Precision Prec>
void SparseMatrixGateImpl<Prec>::update_quantum_state(StateVectorBatched<Prec>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sparse_matrix_gate(this->_target_mask, this->_control_mask, _matrix, states);
}
template <Precision Prec>
std::string SparseMatrixGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SparseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
SCALUQ_DECLARE_CLASS_FOR_PRECISION(SparseMatrixGateImpl)
}  // namespace scaluq::internal
