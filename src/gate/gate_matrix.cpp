#include <scaluq/gate/gate_matrix.hpp>

#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
template <Precision Prec>
DenseMatrixGateImpl<Prec>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                               std::uint64_t control_mask,
                                               const ComplexMatrix& mat,
                                               bool is_unitary)
    : GateBase<Prec>(target_mask, control_mask),
      _matrix(convert_external_matrix_to_internal_matrix<Prec>(mat)),
      _is_unitary(is_unitary) {}
template <Precision Prec>
std::shared_ptr<const GateBase<Prec>> DenseMatrixGateImpl<Prec>::get_inverse() const {
    ComplexMatrix mat_eigen = convert_internal_matrix_to_external_matrix(_matrix);
    ComplexMatrix inv_eigen;
=======
FLOAT_AND_SPACE(Fp, Sp)
DenseMatrixGateImpl<Fp, Sp>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                                 std::uint64_t control_mask,
                                                 const ComplexMatrix<Fp>& mat,
                                                 bool is_unitary)
    : GateBase<Fp, Sp>(target_mask, control_mask),
      _matrix(convert_external_matrix_to_internal_matrix<Fp, Sp>(mat)),
      _is_unitary(is_unitary) {}
FLOAT_AND_SPACE(Fp, Sp)
std::shared_ptr<const GateBase<Fp, Sp>> DenseMatrixGateImpl<Fp, Sp>::get_inverse() const {
    ComplexMatrix<Fp> mat_eigen = convert_internal_matrix_to_external_matrix(_matrix);
    ComplexMatrix<Fp> inv_eigen;
>>>>>>> set-space
    if (_is_unitary) {
        inv_eigen = mat_eigen.adjoint();
    } else {
        // inv_eigen = mat_eigen.inverse().eval();
        throw std::runtime_error("inverse of non-unitary matrix gate is currently not available.");
    }
    return std::make_shared<const DenseMatrixGateImpl>(
        this->_target_mask, this->_control_mask, inv_eigen, _is_unitary);
}
<<<<<<< HEAD
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
=======
FLOAT_AND_SPACE(Fp, Sp)
Matrix<Fp, Sp> DenseMatrixGateImpl<Fp, Sp>::get_matrix_internal() const {
    Matrix<Fp, Sp> ret("return matrix", _matrix.extent(0), _matrix.extent(1));
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
>>>>>>> set-space
    std::ostringstream ss;
    ss << indent << "Gate Type: DenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(DenseMatrixGateImpl)

template <Precision Prec>
SparseMatrixGateImpl<Prec>::SparseMatrixGateImpl(std::uint64_t target_mask,
                                                 std::uint64_t control_mask,
                                                 const SparseComplexMatrix& mat)
    : GateBase<Prec>(target_mask, control_mask),
      _matrix(SparseMatrix<Prec>(mat)),
      num_nnz(mat.nonZeros()) {}
template <Precision Prec>
std::shared_ptr<const GateBase<Prec>> SparseMatrixGateImpl<Prec>::get_inverse() const {
    throw std::runtime_error("inverse of sparse matrix gate is currently not available.");
}
template <Precision Prec>
Matrix<Prec> SparseMatrixGateImpl<Prec>::get_matrix_internal() const {
    Matrix<Prec> ret("return matrix", _matrix._row, _matrix._col);
=======
FLOAT_AND_SPACE_DECLARE_CLASS(DenseMatrixGateImpl)

FLOAT_AND_SPACE(Fp, Sp)
SparseMatrixGateImpl<Fp, Sp>::SparseMatrixGateImpl(std::uint64_t target_mask,
                                                   std::uint64_t control_mask,
                                                   const SparseComplexMatrix<Fp>& mat)
    : GateBase<Fp, Sp>(target_mask, control_mask), _matrix(mat), num_nnz(mat.nonZeros()) {}
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
Matrix<Fp, Sp> SparseMatrixGateImpl<Fp, Sp>::get_matrix_internal() const {
    Matrix<Fp, Sp> ret("return matrix", _matrix._row, _matrix._col);
>>>>>>> set-space
    auto vec = _matrix._values;
    Kokkos::parallel_for(
        vec.size(), KOKKOS_LAMBDA(int i) { ret(vec[i].r, vec[i].c) = vec[i].val; });
    return ret;
}
<<<<<<< HEAD
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
=======
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
>>>>>>> set-space
    std::ostringstream ss;
    ss << indent << "Gate Type: SparseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(SparseMatrixGateImpl)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(SparseMatrixGateImpl)
>>>>>>> set-space
}  // namespace scaluq::internal
