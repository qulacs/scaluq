#include <scaluq/gate/gate_matrix.hpp>

#include "../prec_space.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <>
DenseMatrixGateImpl<Prec, Space>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                                      std::uint64_t control_mask,
                                                      std::uint64_t control_value_mask,
                                                      const ComplexMatrix& mat,
                                                      bool is_unitary)
    : GateBase<Prec, Space>(target_mask, control_mask, control_value_mask),
      _matrix(convert_external_matrix_to_internal_matrix<Prec, Space>(mat)),
      _is_unitary(is_unitary) {}
template <>
std::shared_ptr<const GateBase<Prec, Space>> DenseMatrixGateImpl<Prec, Space>::get_inverse() const {
    ComplexMatrix mat_eigen = convert_internal_matrix_to_external_matrix<Prec, Space>(_matrix);
    ComplexMatrix inv_eigen;
    if (_is_unitary) {
        inv_eigen = mat_eigen.adjoint();
    } else {
        // inv_eigen = mat_eigen.inverse().eval();
        throw std::runtime_error("inverse of non-unitary matrix gate is currently not available.");
    }
    return std::make_shared<const DenseMatrixGateImpl>(
        this->_target_mask, this->_control_mask, this->_control_value_mask, inv_eigen, _is_unitary);
}
template <>
Matrix<Prec, Space> DenseMatrixGateImpl<Prec, Space>::get_matrix_internal() const {
    Matrix<Prec, Space> ret("return matrix", _matrix.extent(0), _matrix.extent(1));
    Kokkos::deep_copy(ret, _matrix);
    return ret;
}
template <>
ComplexMatrix DenseMatrixGateImpl<Prec, Space>::get_matrix() const {
    return convert_internal_matrix_to_external_matrix<Prec, Space>(_matrix);
}
template <>
void DenseMatrixGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    dense_matrix_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _matrix, state_vector);
}
template <>
void DenseMatrixGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    dense_matrix_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _matrix, states);
}
template <>
std::string DenseMatrixGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: DenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class DenseMatrixGateImpl<Prec, Space>;

template <>
SparseMatrixGateImpl<Prec, Space>::SparseMatrixGateImpl(std::uint64_t target_mask,
                                                        std::uint64_t control_mask,
                                                        std::uint64_t control_value_mask,
                                                        const SparseComplexMatrix& mat)
    : GateBase<Prec, Space>(target_mask, control_mask, control_value_mask),
      _matrix(SparseMatrix<Prec, Space>(mat)),
      num_nnz(mat.nonZeros()) {}
template <>
std::shared_ptr<const GateBase<Prec, Space>> SparseMatrixGateImpl<Prec, Space>::get_inverse()
    const {
    throw std::runtime_error("inverse of sparse matrix gate is currently not available.");
}
template <>
Matrix<Prec, Space> SparseMatrixGateImpl<Prec, Space>::get_matrix_internal() const {
    Matrix<Prec, Space> ret("return matrix", _matrix._row, _matrix._col);
    auto vec = _matrix._values;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<SpaceType<Space>>(0, vec.size()),
        KOKKOS_LAMBDA(int i) { ret(vec[i].r, vec[i].c) = vec[i].val; });
    return ret;
}
template <>
ComplexMatrix SparseMatrixGateImpl<Prec, Space>::get_matrix() const {
    return convert_coo_to_external_matrix(_matrix);
}
template <>
void SparseMatrixGateImpl<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state_vector) const {
    this->check_qubit_mask_within_bounds(state_vector);
    sparse_matrix_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _matrix, state_vector);
}
template <>
void SparseMatrixGateImpl<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states) const {
    this->check_qubit_mask_within_bounds(states);
    sparse_matrix_gate(
        this->_target_mask, this->_control_mask, this->_control_value_mask, _matrix, states);
}
template <>
std::string SparseMatrixGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SparseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
template class SparseMatrixGateImpl<Prec, Space>;
}  // namespace scaluq::internal
