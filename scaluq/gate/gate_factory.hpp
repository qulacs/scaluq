#pragma once

#include "gate_matrix.hpp"
#include "gate_pauli.hpp"
#include "gate_probablistic.hpp"
#include "gate_standard.hpp"

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
inline Gate GlobalPhase(double phase, const std::vector<std::uint64_t>& control_qubits = {}) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl>(
        internal::vector_to_mask(control_qubits), phase);
}
inline Gate X(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Y(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::YGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Z(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::ZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate H(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::HGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate S(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Sdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate T(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate Tdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtX(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtXdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtY(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate SqrtYdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate P0(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P0GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate P1(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P1GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
inline Gate RX(std::uint64_t target,
               double angle,
               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RXGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
inline Gate RY(std::uint64_t target,
               double angle,
               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RYGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
inline Gate RZ(std::uint64_t target,
               double angle,
               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
inline Gate U1(std::uint64_t target,
               double lambda,
               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U1GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), lambda);
}
inline Gate U2(std::uint64_t target,
               double phi,
               double lambda,
               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U2GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), phi, lambda);
}
inline Gate U3(std::uint64_t target,
               double theta,
               double phi,
               double lambda,
               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U3GateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), theta, phi, lambda);
}
inline Gate OneTargetMatrix(std::uint64_t target,
                            const std::array<std::array<Complex, 2>, 2>& matrix,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::OneTargetMatrixGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), matrix);
}
inline Gate CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
inline auto& CNot = CX;
inline Gate CZ(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
inline Gate CCX(std::uint64_t control1, std::uint64_t control2, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control1, control2}));
}
inline auto& Toffoli = CCX;
inline auto& CCNot = CCX;
inline Gate Swap(std::uint64_t target1,
                 std::uint64_t target2,
                 const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls));
}
inline Gate TwoTargetMatrix(std::uint64_t target1,
                            std::uint64_t target2,
                            const std::array<std::array<Complex, 4>, 4>& matrix,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TwoTargetMatrixGateImpl>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls), matrix);
}
// まだ
inline Gate Pauli(const PauliOperator& pauli, const std::vector<std::uint64_t>& controls = {}) {
    auto tar = pauli.get_target_qubit_list();
    return internal::GateFactory::create_gate<internal::PauliGateImpl>(
        internal::vector_to_mask(controls), pauli);
}
inline Gate PauliRotation(const PauliOperator& pauli,
                          double angle,
                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl>(
        internal::vector_to_mask(controls), pauli, angle);
}
inline Gate DenseMatrix(const std::vector<std::uint64_t>& targets,
                        const ComplexMatrix& matrix,
                        const std::vector<std::uint64_t>& controls = {}) {
    std::uint64_t nqubits = targets.size();
    std::uint64_t dim = 1ULL << nqubits;
    if (static_cast<std::uint64_t>(matrix.rows()) != dim ||
        static_cast<std::uint64_t>(matrix.cols()) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<std::uint64_t>&, const ComplexMatrix&): matrix "
            "size "
            "must "
            "be 2^{n_qubits} x 2^{n_qubits}.");
    }
    if (targets.size() == 0) return I();
    if (targets.size() == 1) {
        return OneTargetMatrix(targets[0],
                               std::array{std::array{Complex(matrix(0, 0)), Complex(matrix(0, 1))},
                                          std::array{Complex(matrix(1, 0)), Complex(matrix(1, 1))}},
                               controls);
    }
    if (targets.size() == 2) {
        return TwoTargetMatrix(targets[0],
                               targets[1],
                               std::array{std::array{Complex(matrix(0, 0)),
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
                                                     Complex(matrix(3, 3))}},
                               controls);
    }
    throw std::runtime_error(
        "gate::DenseMatrix(const std::vector<std::uint64_t>&, const ComplexMatrix&): DenseMatrix "
        "gate "
        "more "
        "than two qubits is not implemented yet.");
}
inline Gate Probablistic(const std::vector<double>& distribution,
                         const std::vector<Gate>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl>(distribution,
                                                                              gate_list);
}
}  // namespace gate
}  // namespace scaluq
