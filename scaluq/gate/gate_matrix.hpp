#pragma once

#include <ranges>
#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_standard.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point FloatType>
class OneTargetMatrixGateImpl : public GateBase<FloatType> {
    Matrix2x2 _matrix;

public:
    OneTargetMatrixGateImpl(std::uint64_t target_mask,
                            std::uint64_t control_mask,
                            const std::array<std::array<Complex, 2>, 2>& matrix)
        : GateBase<FloatType>(target_mask, control_mask) {
        _matrix[0][0] = matrix[0][0];
        _matrix[0][1] = matrix[0][1];
        _matrix[1][0] = matrix[1][0];
        _matrix[1][1] = matrix[1][1];
    }

    std::array<std::array<Complex, 2>, 2> matrix() const {
        return {_matrix[0][0], _matrix[0][1], _matrix[1][0], _matrix[1][1]};
    }

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const OneTargetMatrixGateImpl>(
            this->_target_mask,
            this->_control_mask,
            std::array<std::array<Complex, 2>, 2>{Kokkos::conj(_matrix[0][0]),
                                                  Kokkos::conj(_matrix[1][0]),
                                                  Kokkos::conj(_matrix[0][1]),
                                                  Kokkos::conj(_matrix[1][1])});
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << this->_matrix[0][0], this->_matrix[0][1], this->_matrix[1][0], this->_matrix[1][1];
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        one_target_dense_matrix_gate(
            this->_target_mask, this->_control_mask, _matrix, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "std::shared_ptr<const GateBase<FloatType>> Type: OneTargetMatrix\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class TwoTargetMatrixGateImpl : public GateBase<FloatType> {
    Matrix4x4 _matrix;

public:
    TwoTargetMatrixGateImpl(std::uint64_t target_mask,
                            std::uint64_t control_mask,
                            const std::array<std::array<Complex, 4>, 4>& matrix)
        : GateBase<FloatType>(target_mask, control_mask) {
        for (std::uint64_t i : std::views::iota(0, 4)) {
            for (std::uint64_t j : std::views::iota(0, 4)) {
                _matrix[i][j] = matrix[i][j];
            }
        }
    }

    std::array<std::array<Complex, 4>, 4> matrix() const {
        std::array<std::array<Complex, 4>, 4> matrix;
        for (std::uint64_t i : std::views::iota(0, 4)) {
            for (std::uint64_t j : std::views::iota(0, 4)) {
                matrix[i][j] = _matrix[i][j];
            }
        }
        return matrix;
    }

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        std::array<std::array<Complex, 4>, 4> matrix_dag;
        for (std::uint64_t i : std::views::iota(0, 4)) {
            for (std::uint64_t j : std::views::iota(0, 4)) {
                matrix_dag[i][j] = Kokkos::conj(_matrix[j][i]);
            }
        }
        return std::make_shared<const TwoTargetMatrixGateImpl>(
            this->_target_mask, this->_control_mask, matrix_dag);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(4, 4);
        mat << this->_matrix[0][0], this->_matrix[0][1], this->_matrix[0][2], this->_matrix[0][3],
            this->_matrix[1][0], this->_matrix[1][1], this->_matrix[1][2], this->_matrix[1][3],
            this->_matrix[2][0], this->_matrix[2][1], this->_matrix[2][2], this->_matrix[2][3],
            this->_matrix[3][0], this->_matrix[3][1], this->_matrix[3][2], this->_matrix[3][3];
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        two_target_dense_matrix_gate(
            this->_target_mask, this->_control_mask, _matrix, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "std::shared_ptr<const GateBase<FloatType>> Type: TwoTargetMatrix\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};
}  // namespace internal

template <std::floating_point FloatType>
using OneTargetMatrixGate = internal::GatePtr<internal::OneTargetMatrixGateImpl<FloatType>>;
template <std::floating_point FloatType>
using TwoTargetMatrixGate = internal::GatePtr<internal::TwoTargetMatrixGateImpl<FloatType>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_gate_matrix_hpp(nb::module_& m) {
    DEF_GATE(OneTargetMatrixGate, "Specific class of one-qubit dense matrix gate.")
        .def("matrix", [](const OneTargetMatrixGate& gate) { return gate->matrix(); });
    DEF_GATE(TwoTargetMatrixGate, "Specific class of two-qubit dense matrix gate.")
        .def("matrix", [](const TwoTargetMatrixGate& gate) { return gate->matrix(); });
}
}  // namespace internal
#endif
}  // namespace scaluq
