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
    auto tar = pauli.target_qubit_list();
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

namespace internal {

template <GateImpl T>
std::string gate_to_string(const GatePtr<T>& obj, std::uint32_t depth = 0) {
    std::ostringstream ss;
    std::string indent(depth * 2, ' ');

    if (obj.gate_type() == GateType::Probablistic) {
        const auto prob_gate = ProbablisticGate(obj);
        const auto distribution = prob_gate->distribution();
        const auto gates = prob_gate->gate_list();
        ss << indent << "Gate Type: Probablistic\n";
        for (std::size_t i = 0; i < distribution.size(); ++i) {
            ss << indent << "  --------------------\n";
            ss << indent << "  Probability: " << distribution[i] << "\n";
            ss << gate_to_string(gates[i], depth + 1) << (i == distribution.size() - 1 ? "" : "\n");
        }
        return ss.str();
    }

    auto targets = internal::mask_to_vector(obj->target_qubit_mask());
    auto controls = internal::mask_to_vector(obj->control_qubit_mask());

    ss << indent << "Gate Type: ";
    switch (obj.gate_type()) {
        case GateType::I:
            ss << "I";
            break;
        case GateType::GlobalPhase:
            ss << "GlobalPhase";
            break;
        case GateType::X:
            ss << "X";
            break;
        case GateType::Y:
            ss << "Y";
            break;
        case GateType::Z:
            ss << "Z";
            break;
        case GateType::H:
            ss << "H";
            break;
        case GateType::S:
            ss << "S";
            break;
        case GateType::Sdag:
            ss << "Sdag";
            break;
        case GateType::T:
            ss << "T";
            break;
        case GateType::Tdag:
            ss << "Tdag";
            break;
        case GateType::SqrtX:
            ss << "SqrtX";
            break;
        case GateType::SqrtXdag:
            ss << "SqrtXdag";
            break;
        case GateType::SqrtY:
            ss << "SqrtY";
            break;
        case GateType::SqrtYdag:
            ss << "SqrtYdag";
            break;
        case GateType::P0:
            ss << "P0";
            break;
        case GateType::P1:
            ss << "P1";
            break;
        case GateType::RX:
            ss << "RX";
            break;
        case GateType::RY:
            ss << "RY";
            break;
        case GateType::RZ:
            ss << "RZ";
            break;
        case GateType::U1:
            ss << "U1";
            break;
        case GateType::U2:
            ss << "U2";
            break;
        case GateType::U3:
            ss << "U3";
            break;
        case GateType::OneTargetMatrix:
            ss << "OneTargetMatrix";
            break;
        case GateType::CX:
            ss << "CX";
            break;
        case GateType::CZ:
            ss << "CZ";
            break;
        case GateType::CCX:
            ss << "CCX";
            break;
        case GateType::Swap:
            ss << "Swap";
            break;
        case GateType::TwoTargetMatrix:
            ss << "TwoTargetMatrix";
            break;
        case GateType::Pauli:
            ss << "Pauli";
            break;
        case GateType::PauliRotation:
            ss << "PauliRotation";
            break;
        case GateType::Unknown:
        default:
            ss << "Unknown";
            break;
    }

    ss << "\n";
    ss << indent << "  Target Qubits: {";
    for (std::uint32_t i = 0; i < targets.size(); ++i)
        ss << targets[i] << (i == targets.size() - 1 ? "" : ", ");
    ss << "}\n";
    ss << indent << "  Control Qubits: {";
    for (std::uint32_t i = 0; i < controls.size(); ++i)
        ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
    ss << "}";

    return ss.str();
}

}  // namespace internal

template <internal::GateImpl T>
std::ostream& operator<<(std::ostream& os, const internal::GatePtr<T>& obj) {
    os << internal::gate_to_string(obj);
    return os;
}

}  // namespace scaluq
