#pragma once

#include "gate_matrix.hpp"
#include "gate_npair_qubit.hpp"
#include "gate_one_control_one_target.hpp"
#include "gate_one_qubit.hpp"
#include "gate_pauli.hpp"
#include "gate_probablistic.hpp"
#include "gate_two_qubit.hpp"
#include "gate_zero_qubit.hpp"

namespace scaluq {
namespace internal {
class GateFactory {
public:
    template <GateImpl T, typename... Args>
    static Gate create_gate(Args... args) {
        return {std::make_shared<T>(args...)};
    }
};
}  // namespace internal
namespace gate {
inline Gate I() { return internal::GateFactory::create_gate<internal::IGateImpl>(); }
inline Gate GlobalPhase(double phase) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl>(phase);
}
inline Gate X(UINT target) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(target);
}
inline Gate Y(UINT target) {
    return internal::GateFactory::create_gate<internal::YGateImpl>(target);
}
inline Gate Z(UINT target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl>(target);
}
inline Gate H(UINT target) {
    return internal::GateFactory::create_gate<internal::HGateImpl>(target);
}
inline Gate S(UINT target) {
    return internal::GateFactory::create_gate<internal::SGateImpl>(target);
}
inline Gate Sdag(UINT target) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl>(target);
}
inline Gate T(UINT target) {
    return internal::GateFactory::create_gate<internal::TGateImpl>(target);
}
inline Gate Tdag(UINT target) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl>(target);
}
inline Gate SqrtX(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl>(target);
}
inline Gate SqrtXdag(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl>(target);
}
inline Gate SqrtY(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl>(target);
}
inline Gate SqrtYdag(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl>(target);
}
inline Gate P0(UINT target) {
    return internal::GateFactory::create_gate<internal::P0GateImpl>(target);
}
inline Gate P1(UINT target) {
    return internal::GateFactory::create_gate<internal::P1GateImpl>(target);
}
inline Gate RX(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RXGateImpl>(target, angle);
}
inline Gate RY(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RYGateImpl>(target, angle);
}
inline Gate RZ(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RZGateImpl>(target, angle);
}
inline Gate U1(UINT target, double lambda) {
    return internal::GateFactory::create_gate<internal::U1GateImpl>(target, lambda);
}
inline Gate U2(UINT target, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U2GateImpl>(target, phi, lambda);
}
inline Gate U3(UINT target, double theta, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U3GateImpl>(target, theta, phi, lambda);
}
inline Gate OneQubitMatrix(UINT target, const std::array<std::array<Complex, 2>, 2>& matrix) {
    return internal::GateFactory::create_gate<internal::OneQubitMatrixGateImpl>(target, matrix);
}
inline Gate CX(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CXGateImpl>(control, target);
}
inline auto& CNot = CX;
inline Gate CZ(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CZGateImpl>(control, target);
}
inline Gate Swap(UINT target1, UINT target2) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl>(target1, target2);
}
inline Gate FusedSwap(UINT qubit_index1, UINT qubit_index2, UINT block_size) {
    return internal::GateFactory::create_gate<internal::FusedSwapGateImpl>(
        qubit_index1, qubit_index2, block_size);
}
inline Gate TwoQubitMatrix(UINT target1,
                           UINT target2,
                           const std::array<std::array<Complex, 4>, 4>& matrix) {
    return internal::GateFactory::create_gate<internal::TwoQubitMatrixGateImpl>(
        target1, target2, matrix);
}
inline Gate Pauli(const PauliOperator& pauli) {
    return internal::GateFactory::create_gate<internal::PauliGateImpl>(pauli);
}
inline Gate PauliRotation(const PauliOperator& pauli, double angle) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl>(pauli, angle);
}
inline Gate DenseMatrix(const Matrix& matrix,
                        const std::vector<UINT>& targets,
                        const std::vector<UINT>& controls) {
    UINT nqubits = targets.size();
    UINT dim = 1ULL << nqubits;
    if (static_cast<UINT>(matrix.extent(0)) != dim || static_cast<UINT>(matrix.extent(1)) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<UINT>&, const ComplexMatrix&): matrix size must "
            "be 2^{n_qubits} x 2^{n_qubits}.");
    }
    if (controls.size() > 0) {
        throw std::runtime_error(
            "gate::DenseMatrix(const Matrix& matrix, const std::vector<UINT>& targets, const "
            "std::vector<UINT>& controls): Control qubit is not implemented.");
    }
    if (targets.size() == 0) return I();
    if (targets.size() == 1) {
        return OneQubitMatrix(targets[0],
                              std::array{std::array{Complex(matrix(0, 0)), Complex(matrix(0, 1))},
                                         std::array{Complex(matrix(1, 0)), Complex(matrix(1, 1))}});
    }
    if (targets.size() == 2) {
        return TwoQubitMatrix(targets[0],
                              targets[1],
                              std::array{
                                  std::array{Complex(matrix(0, 0)),
                                             Complex(matrix(0, 1)),
                                             Complex(matrix(0, 2)),
                                             Complex(matrix(0, 3))},
                                  std::array{Complex(matrix(1, 0)),
                                             Complex(matrix(1, 1)),
                                             Complex(matrix(1, 2)),
                                             Complex(matrix(1, 3))},
                                  std::array{Complex(matrix(2, 0)),
                                             Complex(matrix(2, 1)),
                                             Complex(matrix(2, 2)),
                                             Complex(matrix(2, 3))},
                                  std::array{Complex(matrix(3, 0)),
                                             Complex(matrix(3, 1)),
                                             Complex(matrix(3, 2)),
                                             Complex(matrix(3, 3))},
                              });
    }
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl>(
        matrix, targets, controls);
}
inline Gate SparseMatrix(const CrsMatrix& matrix,
                         const std::vector<UINT>& target_qubit_list,
                         const std::vector<UINT>& control_qubit_list) {
    return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl>(
        matrix, target_qubit_list, control_qubit_list);
}
inline Gate Probablistic(const std::vector<double>& distribution,
                         const std::vector<Gate>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl>(distribution,
                                                                              gate_list);
}
}  // namespace gate
}  // namespace scaluq
