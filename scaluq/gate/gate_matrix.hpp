#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <ranges>
#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_standard.hpp"
#include "update_ops.hpp"

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
    internal::ComplexMatrix<Fp> get_matrix() const override {
        internal::ComplexMatrix<Fp> mat(2, 2);
        mat << this->_matrix[0][0], this->_matrix[0][1], this->_matrix[1][0], this->_matrix[1][1];
        return mat;
    }

    void update_quantum_state(StateVector<Fp>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        one_target_dense_matrix_gate(
            this->_target_mask, this->_control_mask, _matrix, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "std::shared_ptr<const GateBase<Fp>> Type: OneTargetMatrix\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
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
    internal::ComplexMatrix<Fp> get_matrix() const override {
        internal::ComplexMatrix<Fp> mat(4, 4);
        mat << this->_matrix[0][0], this->_matrix[0][1], this->_matrix[0][2], this->_matrix[0][3],
            this->_matrix[1][0], this->_matrix[1][1], this->_matrix[1][2], this->_matrix[1][3],
            this->_matrix[2][0], this->_matrix[2][1], this->_matrix[2][2], this->_matrix[2][3],
            this->_matrix[3][0], this->_matrix[3][1], this->_matrix[3][2], this->_matrix[3][3];
        return mat;
    }

    void update_quantum_state(StateVector<Fp>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        two_target_dense_matrix_gate(
            this->_target_mask, this->_control_mask, _matrix, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "std::shared_ptr<const GateBase<Fp>> Type: TwoTargetMatrix\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point Fp>
class DenseMatrixGateImpl : public GateBase<Fp> {
    Matrix<Fp> _matrix;
    bool _is_unitary;

public:
    DenseMatrixGateImpl(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const ComplexMatrix<Fp>& mat,
                        bool is_unitary = false)
        : GateBase<Fp>(target_mask, control_mask),
          _matrix(convert_external_matrix_to_internal_matrix(mat)),
          _is_unitary(is_unitary) {}

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
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

    Matrix<Fp> get_matrix_internal() const {
        Matrix<Fp> ret("return matrix", _matrix.extent(0), _matrix.extent(1));
        Kokkos::deep_copy<Fp>(ret, _matrix);
        return ret;
    }

    ComplexMatrix<Fp> get_matrix() const override {
        return convert_internal_matrix_to_external_matrix(_matrix);
    }

    void update_quantum_state(StateVector<Fp>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        dense_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: DenseMatrix\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point Fp>
class SparseMatrixGateImpl : public GateBase<Fp> {
    SparseMatrix<Fp> _matrix;
    std::uint64_t num_nnz;

public:
    SparseMatrixGateImpl(std::uint64_t target_mask,
                         std::uint64_t control_mask,
                         const SparseComplexMatrix<Fp>& mat)
        : GateBase<Fp>(target_mask, control_mask),
          _matrix(SparseMatrix(mat)),
          num_nnz(mat.nonZeros()) {}

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        Kokkos::View<SparseValue<Fp>*, Kokkos::HostSpace> vec_h("h_view", num_nnz);
        Kokkos::deep_copy(vec_h, _matrix._values);
        // conversion to Eigen matrix (COO format)
        ComplexMatrix<Fp> eigen_matrix = ComplexMatrix<Fp>::Zero(_matrix._row, _matrix._col);
        for (std::size_t i = 0; i < vec_h.extent(0); i++) {
            eigen_matrix(vec_h(i).r, vec_h(i).c) = vec_h(i).val;
        }
        return std::make_shared<const DenseMatrixGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, eigen_matrix.inverse().eval());
    }

    Matrix<Fp> get_matrix_internal() const {
        Matrix<Fp> ret("return matrix", _matrix._row, _matrix._col);
        auto vec = _matrix._values;
        Kokkos::parallel_for(
            vec.size(), KOKKOS_LAMBDA(int i) { ret(vec[i].r, vec[i].c) = vec[i].val; });
        return ret;
    }

    ComplexMatrix<Fp> get_matrix() const override {
        return convert_coo_to_external_matrix(_matrix);
    }

    SparseComplexMatrix<Fp> get_sparse_matrix() const { return get_matrix().sparseView(); }

    void update_quantum_state(StateVector<Fp>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        sparse_matrix_gate(this->_target_mask, this->_control_mask, _matrix, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SparseMatrix\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
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
