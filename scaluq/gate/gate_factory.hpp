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
                        const std::vector<std::uint64_t>& controls = {},
                        bool is_unitary = false) {
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
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl>(
        internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix, is_unitary);
}
inline Gate SparseMatrix(const std::vector<std::uint64_t>& targets,
                         const SparseComplexMatrix& matrix,
                         const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl>(
        internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix);
}
inline Gate Probablistic(const std::vector<double>& distribution,
                         const std::vector<Gate>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl>(distribution,
                                                                              gate_list);
}
}  // namespace gate
}  // namespace scaluq
