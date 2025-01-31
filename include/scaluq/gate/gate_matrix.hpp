#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <ranges>
#include <vector>

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
<<<<<<< HEAD
template <Precision Prec>
class DenseMatrixGateImpl : public GateBase<Prec> {
    Matrix<Prec> _matrix;
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class DenseMatrixGateImpl : public GateBase<Fp, Sp> {
    Matrix<Fp, Sp> _matrix;
>>>>>>> set-space
    bool _is_unitary;

public:
    DenseMatrixGateImpl(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const ComplexMatrix& mat,
                        bool is_unitary = false);

<<<<<<< HEAD
    std::shared_ptr<const GateBase<Prec>> get_inverse() const override;

    Matrix<Prec> get_matrix_internal() const;
=======
    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override;

    Matrix<Fp, Sp> get_matrix_internal() const;
>>>>>>> set-space

    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "DensetMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"matrix", "Not inplemented yet"}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class SparseMatrixGateImpl : public GateBase<Prec> {
    SparseMatrix<Prec> _matrix;
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class SparseMatrixGateImpl : public GateBase<Fp, Sp> {
    SparseMatrix<Fp, Sp> _matrix;
>>>>>>> set-space
    std::uint64_t num_nnz;

public:
    SparseMatrixGateImpl(std::uint64_t target_mask,
                         std::uint64_t control_mask,
                         const SparseComplexMatrix& mat);

<<<<<<< HEAD
    std::shared_ptr<const GateBase<Prec>> get_inverse() const override;

    Matrix<Prec> get_matrix_internal() const;
=======
    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override;

    Matrix<Fp, Sp> get_matrix_internal() const;
>>>>>>> set-space

    ComplexMatrix get_matrix() const override;

    SparseComplexMatrix get_sparse_matrix() const { return get_matrix().sparseView(); }

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SparseMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"matrix", "Not inplemented yet"}};
    }
};

}  // namespace internal

<<<<<<< HEAD
template <Precision Prec>
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl<Prec>>;
template <Precision Prec>
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_matrix_hpp(nb::module_& m) {
    DEF_GATE(SparseMatrixGate, Prec, "Specific class of sparse matrix gate.")
        .def("matrix", [](const SparseMatrixGate<Prec>& gate) { return gate->get_matrix(); })
        .def("sparse_matrix",
             [](const SparseMatrixGate<Prec>& gate) { return gate->get_sparse_matrix(); });
    DEF_GATE(DenseMatrixGate, Prec, "Specific class of dense matrix gate.")
        .def("matrix", [](const DenseMatrixGate<Prec>& gate) { return gate->get_matrix(); });
=======
template <std::floating_point Fp, ExecutionSpace Sp>
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl<Fp, Sp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp, ExecutionSpace Sp>
void bind_gate_gate_matrix_hpp(nb::module_& m) {
    DEF_GATE(SparseMatrixGate, Fp, "Specific class of sparse matrix gate.")
        .def("matrix", [](const SparseMatrixGate<Fp, Sp>& gate) { return gate->get_matrix(); })
        .def("sparse_matrix",
             [](const SparseMatrixGate<Fp, Sp>& gate) { return gate->get_sparse_matrix(); });
    DEF_GATE(DenseMatrixGate, Fp, "Specific class of dense matrix gate.")
        .def("matrix", [](const DenseMatrixGate<Fp, Sp>& gate) { return gate->get_matrix(); });
>>>>>>> set-space
}
}  // namespace internal
#endif
}  // namespace scaluq
