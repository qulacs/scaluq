#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <ranges>
#include <vector>

#include "../constant.hpp"
#include "gate.hpp"
#include "gate_standard.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp>
class OneTargetMatrixGateImpl : public GateBase<Fp> {
    Matrix2x2<Fp> _matrix;

public:
    OneTargetMatrixGateImpl(std::uint64_t target_mask,
                            std::uint64_t control_mask,
                            const std::array<std::array<Complex<Fp>, 2>, 2>& matrix)
        : GateBase<Fp>(target_mask, control_mask) {
        _matrix[0][0] = matrix[0][0];
        _matrix[0][1] = matrix[0][1];
        _matrix[1][0] = matrix[1][0];
        _matrix[1][1] = matrix[1][1];
    }

    std::array<std::array<Complex<Fp>, 2>, 2> matrix() const {
        return {_matrix[0][0], _matrix[0][1], _matrix[1][0], _matrix[1][1]};
    }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const OneTargetMatrixGateImpl>(
            this->_target_mask,
            this->_control_mask,
            std::array<std::array<Complex<Fp>, 2>, 2>{Kokkos::conj(_matrix[0][0]),
                                                      Kokkos::conj(_matrix[1][0]),
                                                      Kokkos::conj(_matrix[0][1]),
                                                      Kokkos::conj(_matrix[1][1])});
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;
    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class TwoTargetMatrixGateImpl : public GateBase<Fp> {
    Matrix4x4<Fp> _matrix;

public:
    TwoTargetMatrixGateImpl(std::uint64_t target_mask,
                            std::uint64_t control_mask,
                            const std::array<std::array<Complex<Fp>, 4>, 4>& matrix)
        : GateBase<Fp>(target_mask, control_mask) {
        for (std::uint64_t i : std::views::iota(0, 4)) {
            for (std::uint64_t j : std::views::iota(0, 4)) {
                _matrix[i][j] = matrix[i][j];
            }
        }
    }

    std::array<std::array<Complex<Fp>, 4>, 4> matrix() const {
        std::array<std::array<Complex<Fp>, 4>, 4> matrix;
        for (std::uint64_t i : std::views::iota(0, 4)) {
            for (std::uint64_t j : std::views::iota(0, 4)) {
                matrix[i][j] = _matrix[i][j];
            }
        }
        return matrix;
    }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        std::array<std::array<Complex<Fp>, 4>, 4> matrix_dag;
        for (std::uint64_t i : std::views::iota(0, 4)) {
            for (std::uint64_t j : std::views::iota(0, 4)) {
                matrix_dag[i][j] = Kokkos::conj(_matrix[j][i]);
            }
        }
        return std::make_shared<const TwoTargetMatrixGateImpl>(
            this->_target_mask, this->_control_mask, matrix_dag);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
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

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
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

    std::string to_string(const std::string& indent) const override;
};

}  // namespace internal

template <std::floating_point Fp>
using OneTargetMatrixGate = internal::GatePtr<internal::OneTargetMatrixGateImpl<Fp>>;
template <std::floating_point Fp>
using TwoTargetMatrixGate = internal::GatePtr<internal::TwoTargetMatrixGateImpl<Fp>>;
template <std::floating_point Fp>
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl<Fp>>;
template <std::floating_point Fp>
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_gate_matrix_hpp(nb::module_& m) {
    DEF_GATE(OneTargetMatrixGate<double>, "Specific class of one-qubit dense matrix gate.")
        .def("matrix", [](const OneTargetMatrixGate<double>& gate) { return gate->matrix(); });
    DEF_GATE(TwoTargetMatrixGate<double>, "Specific class of two-qubit dense matrix gate.")
        .def("matrix", [](const TwoTargetMatrixGate<double>& gate) { return gate->matrix(); });
    DEF_GATE(SparseMatrixGate<double>, "Specific class of sparse matrix gate.")
        .def("matrix", [](const SparseMatrixGate<double>& gate) { return gate->get_matrix(); })
        .def("sparse_matrix",
             [](const SparseMatrixGate<double>& gate) { return gate->get_sparse_matrix(); });
    DEF_GATE(DenseMatrixGate<double>, "Specific class of dense matrix gate.")
        .def("matrix", [](const DenseMatrixGate<double>& gate) { return gate->get_matrix(); });
}
}  // namespace internal
#endif
}  // namespace scaluq
