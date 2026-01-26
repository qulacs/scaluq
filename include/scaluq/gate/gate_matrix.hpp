#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <ranges>
#include <vector>

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec, ExecutionSpace Space>
class DenseMatrixGateImpl : public GateBase<Prec> {
    Matrix<Prec, Space> _matrix;
    bool _is_unitary;

public:
    DenseMatrixGateImpl(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const ComplexMatrix& mat,
                        bool is_unitary = false);

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override;

    Matrix<Prec, Space> get_matrix_internal() const;

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::HostSerial>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::HostSerial>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "DenseMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"matrix", this->get_matrix()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SparseMatrixGateImpl : public GateBase<Prec> {
    SparseMatrix<Prec, Space> _matrix;
    std::uint64_t num_nnz;

public:
    SparseMatrixGateImpl(std::uint64_t target_mask,
                         std::uint64_t control_mask,
                         std::uint64_t control_value_mask,
                         const SparseComplexMatrix& mat);

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override;

    Matrix<Prec, Space> get_matrix_internal() const;

    ComplexMatrix get_matrix() const override;

    SparseComplexMatrix get_sparse_matrix() const { return get_matrix().sparseView(); }

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::HostSerial>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::HostSerial>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SparseMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"matrix", this->get_sparse_matrix()}};
    }
};

}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl<Prec, Space>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_matrix_hpp(nb::module_& m, nb::class_<Gate<Prec>>& gate_base_def) {
    bind_specific_gate<SparseMatrixGate<Prec, Space>, Prec>(
        m, gate_base_def, "SparseMatrixGate", "Specific class of sparse matrix gate.");
    bind_specific_gate<DenseMatrixGate<Prec, Space>, Prec>(
        m, gate_base_def, "DenseMatrixGate", "Specific class of dense matrix gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
