#include <scaluq/gate/gate_matrix.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec, ExecutionSpace Space>
OneTargetDenseMatrixGateImpl<Prec, Space>::OneTargetDenseMatrixGateImpl(
    std::uint64_t target_mask,
    std::uint64_t control_mask,
    std::uint64_t control_value_mask,
    const ComplexMatrix& mat,
    bool is_unitary)
    : GateBase<Prec>(target_mask, control_mask, control_value_mask),
      _is_unitary(is_unitary) {
    for (std::uint64_t i = 0; i < 2; ++i) {
        for (std::uint64_t j = 0; j < 2; ++j) {
            _matrix[i][j] = mat(i, j);
        }
    }
}
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const GateBase<Prec>> OneTargetDenseMatrixGateImpl<Prec, Space>::get_inverse()
    const {
    ComplexMatrix mat_eigen = get_matrix();
    ComplexMatrix inv_eigen;
    if (_is_unitary) {
        inv_eigen = mat_eigen.adjoint();
    } else {
        throw std::runtime_error("inverse of non-unitary matrix gate is currently not available.");
    }
    return std::make_shared<const OneTargetDenseMatrixGateImpl>(
        this->_target_mask, this->_control_mask, this->_control_value_mask, inv_eigen, _is_unitary);
}
template <Precision Prec, ExecutionSpace Space>
ComplexMatrix OneTargetDenseMatrixGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix ret(2, 2);
    for (std::uint64_t i = 0; i < 2; ++i) {
        for (std::uint64_t j = 0; j < 2; ++j) {
            ret(i, j) = static_cast<StdComplex>(_matrix[i][j]);
        }
    }
    return ret;
}
template <Precision Prec, ExecutionSpace Space>
std::string OneTargetDenseMatrixGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: OneTargetDenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}

#define DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE(ContextClass, state_member, TargetSpace)  \
    template <Precision Prec, ExecutionSpace GateSpace>                                      \
    void OneTargetDenseMatrixGateImpl<Prec, GateSpace>::update_quantum_state(                \
        ContextClass<Prec, TargetSpace>& context) const {                                    \
        if constexpr (GateSpace == TargetSpace) {                                            \
            this->check_qubit_mask_within_bounds(context.state_member);                      \
            one_target_dense_matrix_gate(this->_target_mask,                                 \
                                         this->_control_mask,                                \
                                         this->_control_value_mask,                          \
                                         _matrix,                                            \
                                         context.state_member);                              \
        } else {                                                                             \
            throw std::runtime_error(                                                        \
                "Error: OneTargetDenseMatrixGateImpl::update_quantum_state(" #ContextClass   \
                "& state_vector): Trying to run on " #TargetSpace                            \
                " execution space, but the gate is defined on different execution space.");  \
        }                                                                                    \
    }
DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Host)
DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Host)
DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::HostSerial)
DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Default)
DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Default)
#endif
#undef DEFINE_ONE_TARGET_DENSE_MATRIX_GATE_UPDATE
template class OneTargetDenseMatrixGateImpl<Prec, Space>;

template <Precision Prec, ExecutionSpace Space>
TwoTargetDenseMatrixGateImpl<Prec, Space>::TwoTargetDenseMatrixGateImpl(
    std::uint64_t target_mask,
    std::uint64_t control_mask,
    std::uint64_t control_value_mask,
    const ComplexMatrix& mat,
    bool is_unitary)
    : GateBase<Prec>(target_mask, control_mask, control_value_mask),
      _is_unitary(is_unitary) {
    for (std::uint64_t i = 0; i < 4; ++i) {
        for (std::uint64_t j = 0; j < 4; ++j) {
            _matrix[i][j] = mat(i, j);
        }
    }
}
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const GateBase<Prec>> TwoTargetDenseMatrixGateImpl<Prec, Space>::get_inverse()
    const {
    ComplexMatrix mat_eigen = get_matrix();
    ComplexMatrix inv_eigen;
    if (_is_unitary) {
        inv_eigen = mat_eigen.adjoint();
    } else {
        throw std::runtime_error("inverse of non-unitary matrix gate is currently not available.");
    }
    return std::make_shared<const TwoTargetDenseMatrixGateImpl>(
        this->_target_mask, this->_control_mask, this->_control_value_mask, inv_eigen, _is_unitary);
}
template <Precision Prec, ExecutionSpace Space>
ComplexMatrix TwoTargetDenseMatrixGateImpl<Prec, Space>::get_matrix() const {
    ComplexMatrix ret(4, 4);
    for (std::uint64_t i = 0; i < 4; ++i) {
        for (std::uint64_t j = 0; j < 4; ++j) {
            ret(i, j) = static_cast<StdComplex>(_matrix[i][j]);
        }
    }
    return ret;
}
template <Precision Prec, ExecutionSpace Space>
std::string TwoTargetDenseMatrixGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: TwoTargetDenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}

#define DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE(ContextClass, state_member, TargetSpace)  \
    template <Precision Prec, ExecutionSpace GateSpace>                                      \
    void TwoTargetDenseMatrixGateImpl<Prec, GateSpace>::update_quantum_state(                \
        ContextClass<Prec, TargetSpace>& context) const {                                    \
        if constexpr (GateSpace == TargetSpace) {                                            \
            this->check_qubit_mask_within_bounds(context.state_member);                      \
            two_target_dense_matrix_gate(this->_target_mask,                                 \
                                         this->_control_mask,                                \
                                         this->_control_value_mask,                          \
                                         _matrix,                                            \
                                         context.state_member);                              \
        } else {                                                                             \
            throw std::runtime_error(                                                        \
                "Error: TwoTargetDenseMatrixGateImpl::update_quantum_state(" #ContextClass   \
                "& state_vector): Trying to run on " #TargetSpace                            \
                " execution space, but the gate is defined on different execution space.");  \
        }                                                                                    \
    }
DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Host)
DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Host)
DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::HostSerial)
DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Default)
DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Default)
#endif
#undef DEFINE_TWO_TARGET_DENSE_MATRIX_GATE_UPDATE
template class TwoTargetDenseMatrixGateImpl<Prec, Space>;

template <Precision Prec, ExecutionSpace Space>
DenseMatrixGateImpl<Prec, Space>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                                      std::uint64_t control_mask,
                                                      std::uint64_t control_value_mask,
                                                      const ComplexMatrix& mat,
                                                      bool is_unitary)
    : GateBase<Prec>(target_mask, control_mask, control_value_mask),
      _matrix(convert_external_matrix_to_internal_matrix<Prec, Space>(mat)),
      _is_unitary(is_unitary) {}
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const GateBase<Prec>> DenseMatrixGateImpl<Prec, Space>::get_inverse() const {
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
template <Precision Prec, ExecutionSpace Space>
Matrix<Prec, Space> DenseMatrixGateImpl<Prec, Space>::get_matrix_internal() const {
    Matrix<Prec, Space> ret("return matrix", _matrix.extent(0), _matrix.extent(1));
    Kokkos::deep_copy(ret, _matrix);
    return ret;
}
template <Precision Prec, ExecutionSpace Space>
ComplexMatrix DenseMatrixGateImpl<Prec, Space>::get_matrix() const {
    return convert_internal_matrix_to_external_matrix<Prec, Space>(_matrix);
}
template <Precision Prec, ExecutionSpace Space>
std::string DenseMatrixGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: DenseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_DENSE_MATRIX_GATE_UPDATE(ContextClass, state_member, TargetSpace)            \
    template <Precision Prec, ExecutionSpace GateSpace>                                     \
    void DenseMatrixGateImpl<Prec, GateSpace>::update_quantum_state(                        \
        ContextClass<Prec, TargetSpace>& context) const {                                    \
        if constexpr (GateSpace == TargetSpace) {                                           \
            this->check_qubit_mask_within_bounds(context.state_member);                     \
            dense_matrix_gate(this->_target_mask,                                           \
                              this->_control_mask,                                          \
                              this->_control_value_mask,                                    \
                              _matrix,                                                      \
                              context.state_member);                                        \
        } else {                                                                            \
            throw std::runtime_error(                                                       \
                "Error: DenseMatrixGateImpl::update_quantum_state(" #ContextClass           \
                "& state_vector): Trying to run on " #TargetSpace                           \
                " execution space, but the gate is defined on different execution space."); \
        }                                                                                   \
    }
DEFINE_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Host)
DEFINE_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Host)
DEFINE_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::HostSerial)
DEFINE_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_DENSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Default)
DEFINE_DENSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Default)
#endif
#undef DEFINE_DENSE_MATRIX_GATE_UPDATE
template class DenseMatrixGateImpl<Prec, Space>;

template <Precision Prec, ExecutionSpace Space>
SparseMatrixGateImpl<Prec, Space>::SparseMatrixGateImpl(std::uint64_t target_mask,
                                                        std::uint64_t control_mask,
                                                        std::uint64_t control_value_mask,
                                                        const SparseComplexMatrix& mat)
    : GateBase<Prec>(target_mask, control_mask, control_value_mask),
      _matrix(SparseMatrix<Prec, Space>(mat)),
      num_nnz(mat.nonZeros()) {}
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const GateBase<Prec>> SparseMatrixGateImpl<Prec, Space>::get_inverse() const {
    throw std::runtime_error("inverse of sparse matrix gate is currently not available.");
}
template <Precision Prec, ExecutionSpace Space>
Matrix<Prec, Space> SparseMatrixGateImpl<Prec, Space>::get_matrix_internal() const {
    Matrix<Prec, Space> ret("return matrix", _matrix._rows, _matrix._cols);
    auto _row_ptr = _matrix._row_ptr;
    auto _col_idx = _matrix._col_idx;
    auto _vals = _matrix._vals;
    Kokkos::parallel_for(
        "get_matrix_internal",
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), _matrix._rows, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t r = team.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, _row_ptr[r], _row_ptr[r + 1]),
                                 [&](std::uint64_t idx) { ret(r, _col_idx[idx]) = _vals[idx]; });
        });
    return ret;
}
template <Precision Prec, ExecutionSpace Space>
ComplexMatrix SparseMatrixGateImpl<Prec, Space>::get_matrix() const {
    return convert_csr_to_external_matrix(_matrix);
}
template <Precision Prec, ExecutionSpace Space>
std::string SparseMatrixGateImpl<Prec, Space>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SparseMatrix\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}

#define DEFINE_SPARSE_MATRIX_GATE_UPDATE(ContextClass, state_member, TargetSpace)           \
    template <Precision Prec, ExecutionSpace GateSpace>                                     \
    void SparseMatrixGateImpl<Prec, GateSpace>::update_quantum_state(                       \
        ContextClass<Prec, TargetSpace>& context) const {                                    \
        if constexpr (GateSpace == TargetSpace) {                                           \
            this->check_qubit_mask_within_bounds(context.state_member);                     \
            sparse_matrix_gate(this->_target_mask,                                          \
                               this->_control_mask,                                         \
                               this->_control_value_mask,                                   \
                               _matrix,                                                     \
                               context.state_member);                                       \
        } else {                                                                            \
            throw std::runtime_error(                                                       \
                "Error: SparseMatrixGateImpl::update_quantum_state(" #ContextClass          \
                "& state_vector): Trying to run on " #TargetSpace                           \
                " execution space, but the gate is defined on different execution space."); \
        }                                                                                   \
    }
DEFINE_SPARSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Host)
DEFINE_SPARSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Host)
DEFINE_SPARSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::HostSerial)
DEFINE_SPARSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_SPARSE_MATRIX_GATE_UPDATE(ExecutionContext, state, ExecutionSpace::Default)
DEFINE_SPARSE_MATRIX_GATE_UPDATE(ExecutionContextBatched, states, ExecutionSpace::Default)
#endif
#undef DEFINE_SPARSE_MATRIX_GATE_UPDATE
template class SparseMatrixGateImpl<Prec, Space>;

template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const DenseMatrixGateImpl<Prec, Space>>
GetGateFromJson<DenseMatrixGateImpl<Prec, Space>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const DenseMatrixGateImpl<Prec, Space>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        j.at("matrix").get<ComplexMatrix>());
}
template struct GetGateFromJson<DenseMatrixGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const OneTargetDenseMatrixGateImpl<Prec, Space>>
GetGateFromJson<OneTargetDenseMatrixGateImpl<Prec, Space>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const OneTargetDenseMatrixGateImpl<Prec, Space>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        j.at("matrix").get<ComplexMatrix>());
}
template struct GetGateFromJson<OneTargetDenseMatrixGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const TwoTargetDenseMatrixGateImpl<Prec, Space>>
GetGateFromJson<TwoTargetDenseMatrixGateImpl<Prec, Space>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const TwoTargetDenseMatrixGateImpl<Prec, Space>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        j.at("matrix").get<ComplexMatrix>());
}
template struct GetGateFromJson<TwoTargetDenseMatrixGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const SparseMatrixGateImpl<Prec, Space>>
GetGateFromJson<SparseMatrixGateImpl<Prec, Space>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const SparseMatrixGateImpl<Prec, Space>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        j.at("matrix").get<SparseComplexMatrix>());
}
template struct GetGateFromJson<SparseMatrixGateImpl<Prec, Space>>;
}  // namespace scaluq::internal
