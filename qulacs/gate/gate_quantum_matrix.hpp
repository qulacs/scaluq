#pragma once

#include <ranges>
#include <vector>

#include "constant.hpp"
#include "gate.hpp"
#include "gate_one_qubit.hpp"
#include "gate_two_qubit.hpp"
#include "update_ops.hpp"

namespace qulacs {
namespace internal {
class U1GateImpl : public OneQubitGateBase {
    double _lambda;
    matrix_2_2 _matrix;

public:
    U1GateImpl(UINT target, double lambda) : OneQubitGateBase(target), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(0, 0, lambda);
    };

    double lambda() const { return _lambda; }

    Gate copy() const override { return std::make_shared<U1GateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<U1GateImpl>(_target, -_lambda); }

    void update_quantum_state(StateVector& state_vector) const override;
};
class U2GateImpl : public OneQubitGateBase {
    double _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U2GateImpl(UINT target, double phi, double lambda)
        : OneQubitGateBase(target), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(PI() / 2.0, phi, lambda);
    };

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate copy() const override { return std::make_shared<U2GateImpl>(*this); }
    Gate get_inverse() const override {
        return std::make_shared<U2GateImpl>(_target, -_lambda - PI(), -_phi + PI());
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class U3GateImpl : public OneQubitGateBase {
    double _theta, _phi, _lambda;
    matrix_2_2 _matrix;

public:
    U3GateImpl(UINT target, double theta, double phi, double lambda)
        : OneQubitGateBase(target), _theta(theta), _phi(phi), _lambda(lambda) {
        _matrix = get_IBMQ_matrix(theta, phi, lambda);
    };

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate copy() const override { return std::make_shared<U3GateImpl>(*this); }
    Gate get_inverse() const override {
        return std::make_shared<U3GateImpl>(_target, -_theta, -_lambda, -_phi);
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class OneQubitMatrixGateImpl : public OneQubitGateBase {
    matrix_2_2 _matrix;

public:
    OneQubitMatrixGateImpl(UINT target, const std::array<std::array<Complex, 2>, 2>& matrix)
        : OneQubitGateBase(target) {
        _matrix.val[0][0] = matrix[0][0];
        _matrix.val[0][1] = matrix[0][1];
        _matrix.val[1][0] = matrix[1][0];
        _matrix.val[1][1] = matrix[1][1];
    }

    std::array<std::array<Complex, 2>, 2> matrix() {
        return {_matrix.val[0][0], _matrix.val[0][1], _matrix.val[1][0], _matrix.val[1][1]};
    }

    Gate copy() const override { return std::make_shared<OneQubitMatrixGateImpl>(*this); }
    Gate get_inverse() const override {
        return std::make_shared<OneQubitMatrixGateImpl>(
            _target,
            std::array<std::array<Complex, 2>, 2>{Kokkos::conj(_matrix.val[0][0]),
                                                  Kokkos::conj(_matrix.val[1][0]),
                                                  Kokkos::conj(_matrix.val[0][1]),
                                                  Kokkos::conj(_matrix.val[1][1])});
    }

    void update_quantum_state(StateVector& state_vector) const override;
};

class TwoQubitMatrixGateImpl : public TwoQubitGateBase {
    matrix_4_4 _matrix;

public:
    TwoQubitMatrixGateImpl(UINT target1,
                           UINT target2,
                           const std::array<std::array<Complex, 4>, 4>& matrix)
        : TwoQubitGateBase(target1, target2) {
        for (UINT i : std::views::iota(4)) {
            for (UINT j : std::views::iota(4)) {
                _matrix.val[i][j] = matrix[i][j];
            }
        }
    }

    std::array<std::array<Complex, 4>, 4> matrix() {
        std::array<std::array<Complex, 4>, 4> matrix;
        for (UINT i : std::views::iota(4)) {
            for (UINT j : std::views::iota(4)) {
                matrix[i][j] = _matrix.val[i][j];
            }
        }
        return matrix;
    }

    Gate copy() const override { return std::make_shared<TwoQubitMatrixGateImpl>(*this); }
    Gate get_inverse() const override {
        std::array<std::array<Complex, 4>, 4> matrix_dag;
        for (UINT i : std::views::iota(4)) {
            for (UINT j : std::views::iota(4)) {
                matrix_dag[i][j] = Kokkos::conj(_matrix.val[j][i]);
            }
        }
        return std::make_shared<TwoQubitMatrixGateImpl>(_target1, _target2, matrix_dag);
    }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using U1Gate = internal::GatePtr<internal::U1GateImpl>;
using U2Gate = internal::GatePtr<internal::U2GateImpl>;
using U3Gate = internal::GatePtr<internal::U3GateImpl>;
using OneQubitMatrixGate = internal::GatePtr<internal::OneQubitMatrixGateImpl>;
using TwoQubitMatrixGate = internal::GatePtr<internal::TwoQubitMatrixGateImpl>;
}  // namespace qulacs
