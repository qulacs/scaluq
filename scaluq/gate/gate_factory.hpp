#pragma once

#include "gate_matrix.hpp"
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
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {
inline Gate I() { return internal::GateFactory::create_gate<internal::IGateImpl>(); }
inline Gate GlobalPhase(const std::vector<UINT>& control_qubits, double phase) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl>(
        internal::vector_to_mask(control_qubits), phase);
}
inline Gate X(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Y(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::YGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Z(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::ZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate H(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::HGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate S(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::SGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Sdag(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate T(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::TGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Tdag(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtX(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtXdag(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtY(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtYdag(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate P0(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::P0GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate P1(UINT target, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::P1GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate RX(UINT target, const std::vector<UINT>& controls, double angle) {
    return internal::GateFactory::create_gate<internal::RXGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
inline Gate RY(UINT target, const std::vector<UINT>& controls, double angle) {
    return internal::GateFactory::create_gate<internal::RYGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
inline Gate RZ(UINT target, const std::vector<UINT>& controls, double angle) {
    return internal::GateFactory::create_gate<internal::RZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
inline Gate U1(UINT target, const std::vector<UINT>& controls, double lambda) {
    return internal::GateFactory::create_gate<internal::U1GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), lambda);
}
inline Gate U2(UINT target, const std::vector<UINT>& controls, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U2GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), phi, lambda);
}
inline Gate U3(
    UINT target, const std::vector<UINT>& controls, double theta, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U3GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), theta, phi, lambda);
}
inline Gate OneQubitMatrix(UINT target,
                           const std::vector<UINT>& controls,
                           const std::array<std::array<Complex, 2>, 2>& matrix) {
    return internal::GateFactory::create_gate<internal::OneQubitMatrixGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), matrix);
}
inline Gate CX(UINT target, UINT control) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
inline auto& CNot = CX;
inline Gate CZ(UINT target, UINT control) {
    return internal::GateFactory::create_gate<internal::ZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
inline Gate Toffoli(UINT target, UINT control1, UINT control2) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control1, control2}));
}
inline auto& CCNot = Toffoli;
inline Gate Swap(UINT target1, UINT target2, const std::vector<UINT>& controls) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls));
}
inline Gate TwoQubitMatrix(UINT target1,
                           UINT target2,
                           const std::vector<UINT>& controls,
                           const std::array<std::array<Complex, 4>, 4>& matrix) {
    return internal::GateFactory::create_gate<internal::TwoQubitMatrixGateImpl>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls), matrix);
}
// まだ
inline Gate Pauli(const std::vector<UINT>& controls, const PauliOperator& pauli) {
    return internal::GateFactory::create_gate<internal::PauliGateImpl>(
        internal::vector_to_mask(controls), pauli);
}
inline Gate PauliRotation(const std::vector<UINT>& controls,
                          const PauliOperator& pauli,
                          double angle) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl>(
        internal::vector_to_mask(controls), pauli, angle);
}
inline Gate DenseMatrix(const std::vector<UINT>& targets,
                        const std::vector<UINT>& controls,
                        const ComplexMatrix& matrix) {
    UINT nqubits = targets.size();
    UINT dim = 1ULL << nqubits;
    if (static_cast<UINT>(matrix.rows()) != dim || static_cast<UINT>(matrix.cols()) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<UINT>&, const ComplexMatrix&): matrix size "
            "must be 2^{n_qubits} x 2^{n_qubits}.");
    }
    if (targets.size() == 0) return I();
    if (targets.size() == 1) {
        return OneQubitMatrix(targets[0],
                              controls,
                              std::array{std::array{Complex(matrix(0, 0)), Complex(matrix(0, 1))},
                                         std::array{Complex(matrix(1, 0)), Complex(matrix(1, 1))}});
    }
    if (targets.size() == 2) {
        return TwoQubitMatrix(targets[0],
                              targets[1],
                              controls,
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
    throw std::runtime_error(
        "gate::DenseMatrix(const std::vector<UINT>&, const ComplexMatrix&): DenseMatrix gate "
        "more than two qubits is not implemented yet.");
}
inline Gate Probablistic(const std::vector<double>& distribution,
                         const std::vector<Gate>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl>(distribution,
                                                                              gate_list);
}
}  // namespace gate
}  // namespace scaluq
