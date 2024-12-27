#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <ranges>
#include <vector>

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
template <FloatingPoint Fp>
class DenseMatrixGateImpl : public GateBase<Fp> {
    Matrix<Fp> _matrix;
    bool _is_unitary;

public:
    DenseMatrixGateImpl(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const ComplexMatrix<Fp>& mat,
                        bool is_unitary = false);

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override;

    Matrix<Fp> get_matrix_internal() const;

    ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "DensetMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"matrix", "Not inplemented yet"}};
    }
};

template <FloatingPoint Fp>
class SparseMatrixGateImpl : public GateBase<Fp> {
    SparseMatrix<Fp> _matrix;
    std::uint64_t num_nnz;

public:
    SparseMatrixGateImpl(std::uint64_t target_mask,
                         std::uint64_t control_mask,
                         const SparseComplexMatrix<Fp>& mat);

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override;

    Matrix<Fp> get_matrix_internal() const;

    ComplexMatrix<Fp> get_matrix() const override;

    SparseComplexMatrix<Fp> get_sparse_matrix() const { return get_matrix().sparseView(); }

    void update_quantum_state(StateVector<Fp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SparseMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"matrix", "Not inplemented yet"}};
    }
};

}  // namespace internal

template <FloatingPoint Fp>
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl<Fp>>;
template <FloatingPoint Fp>
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <FloatingPoint Fp>
void bind_gate_gate_matrix_hpp(nb::module_& m) {
    DEF_GATE(SparseMatrixGate, Fp, "Specific class of sparse matrix gate.")
        .def("matrix", [](const SparseMatrixGate<Fp>& gate) { return gate->get_matrix(); })
        .def("sparse_matrix",
             [](const SparseMatrixGate<Fp>& gate) { return gate->get_sparse_matrix(); });
    DEF_GATE(DenseMatrixGate, Fp, "Specific class of dense matrix gate.")
        .def("matrix", [](const DenseMatrixGate<Fp>& gate) { return gate->get_matrix(); });
}
}  // namespace internal
#endif
}  // namespace scaluq
