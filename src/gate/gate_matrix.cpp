#include <scaluq/gate/gate_matrix.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec, ExecutionSpace Space>
DenseMatrixGateImpl<Prec, Space>::DenseMatrixGateImpl(std::uint64_t target_mask,
                                                      std::uint64_t control_mask,
                                                      std::uint64_t control_value_mask,
                                                      const ComplexMatrix& mat,
                                                      bool is_unitary)
    : GateBase<Prec>(target_mask, control_mask, control_value_mask), _is_unitary(is_unitary) {
    if (mat.rows() == 1 && mat.cols() == 1) {
        _matrix = Complex<Prec>(static_cast<Float<Prec>>(mat(0, 0).real()),
                                static_cast<Float<Prec>>(mat(0, 0).imag()));
    } else if (mat.rows() == 2 && mat.cols() == 2) {
        Matrix2x2<Prec> matrix;
        for (std::uint64_t i = 0; i < 2; ++i) {
            for (std::uint64_t j = 0; j < 2; ++j) {
                matrix[i][j] = Complex<Prec>(static_cast<Float<Prec>>(mat(i, j).real()),
                                             static_cast<Float<Prec>>(mat(i, j).imag()));
            }
        }
        _matrix = matrix;
    } else if (mat.rows() == 4 && mat.cols() == 4) {
        Matrix4x4<Prec> matrix;
        for (std::uint64_t i = 0; i < 4; ++i) {
            for (std::uint64_t j = 0; j < 4; ++j) {
                matrix[i][j] = Complex<Prec>(static_cast<Float<Prec>>(mat(i, j).real()),
                                             static_cast<Float<Prec>>(mat(i, j).imag()));
            }
        }
        _matrix = matrix;
    } else {
        _matrix = convert_external_matrix_to_internal_matrix<Prec, Space>(mat);
    }
}
template <Precision Prec, ExecutionSpace Space>
std::shared_ptr<const GateBase<Prec>> DenseMatrixGateImpl<Prec, Space>::get_inverse() const {
    ComplexMatrix mat_eigen = get_matrix();
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
    return std::visit(
        [](const auto& matrix) -> Matrix<Prec, Space> {
            using T = std::decay_t<decltype(matrix)>;
            if constexpr (std::is_same_v<T, Matrix<Prec, Space>>) {
                Matrix<Prec, Space> ret("return matrix", matrix.extent(0), matrix.extent(1));
                Kokkos::deep_copy(ret, matrix);
                return ret;
            } else {
                ComplexMatrix external;
                if constexpr (std::is_same_v<T, Complex<Prec>>) {
                    external.resize(1, 1);
                    external(0, 0) = static_cast<StdComplex>(matrix);
                } else {
                    const std::uint64_t dim = matrix.size();
                    external.resize(dim, dim);
                    for (std::uint64_t i = 0; i < dim; ++i) {
                        for (std::uint64_t j = 0; j < dim; ++j) {
                            external(i, j) = static_cast<StdComplex>(matrix[i][j]);
                        }
                    }
                }
                return convert_external_matrix_to_internal_matrix<Prec, Space>(external);
            }
        },
        _matrix);
}
template <Precision Prec, ExecutionSpace Space>
ComplexMatrix DenseMatrixGateImpl<Prec, Space>::get_matrix() const {
    return std::visit(
        [](const auto& matrix) -> ComplexMatrix {
            using T = std::decay_t<decltype(matrix)>;
            if constexpr (std::is_same_v<T, Matrix<Prec, Space>>) {
                return convert_internal_matrix_to_external_matrix<Prec, Space>(matrix);
            } else if constexpr (std::is_same_v<T, Complex<Prec>>) {
                ComplexMatrix ret(1, 1);
                ret(0, 0) = static_cast<StdComplex>(matrix);
                return ret;
            } else {
                const std::uint64_t dim = matrix.size();
                ComplexMatrix ret(dim, dim);
                for (std::uint64_t i = 0; i < dim; ++i) {
                    for (std::uint64_t j = 0; j < dim; ++j) {
                        ret(i, j) = static_cast<StdComplex>(matrix[i][j]);
                    }
                }
                return ret;
            }
        },
        _matrix);
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
        ContextClass<Prec, TargetSpace>& context) const {                                   \
        if constexpr (GateSpace == TargetSpace) {                                           \
            this->check_qubit_mask_within_bounds(context.state_member);                     \
            std::visit(                                                                     \
                [&](const auto& matrix) {                                                   \
                    using T = std::decay_t<decltype(matrix)>;                               \
                    if constexpr (std::is_same_v<T, Complex<Prec>>) {                       \
                        zero_target_dense_matrix_gate(this->_control_mask,                  \
                                                      this->_control_value_mask,            \
                                                      matrix,                               \
                                                      context.state_member);                \
                    } else if constexpr (std::is_same_v<T, Matrix2x2<Prec>>) {              \
                        one_target_dense_matrix_gate(this->_target_mask,                    \
                                                     this->_control_mask,                   \
                                                     this->_control_value_mask,             \
                                                     matrix,                                \
                                                     context.state_member);                 \
                    } else if constexpr (std::is_same_v<T, Matrix4x4<Prec>>) {              \
                        two_target_dense_matrix_gate(this->_target_mask,                    \
                                                     this->_control_mask,                   \
                                                     this->_control_value_mask,             \
                                                     matrix,                                \
                                                     context.state_member);                 \
                    } else {                                                                \
                        multi_dense_matrix_gate(this->_target_mask,                         \
                                                this->_control_mask,                        \
                                                this->_control_value_mask,                  \
                                                matrix,                                     \
                                                context.state_member);                      \
                    }                                                                       \
                },                                                                          \
                _matrix);                                                                   \
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
        ContextClass<Prec, TargetSpace>& context) const {                                   \
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
