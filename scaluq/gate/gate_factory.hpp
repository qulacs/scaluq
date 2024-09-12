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
                        const internal::ComplexMatrix& matrix,
                        const std::vector<std::uint64_t>& controls = {},
                        bool is_unitary = false) {
    std::uint64_t nqubits = targets.size();
    std::uint64_t dim = 1ULL << nqubits;
    if (static_cast<std::uint64_t>(matrix.rows()) != dim ||
        static_cast<std::uint64_t>(matrix.cols()) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<std::uint64_t>&, const internal::ComplexMatrix&): "
            "matrix size must be 2^{n_qubits} x 2^{n_qubits}.");
    }
    if (targets.size() == 0) return I();
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl>(
        internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix, is_unitary);
}
inline Gate SparseMatrix(const std::vector<std::uint64_t>& targets,
                         const internal::SparseComplexMatrix& matrix,
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

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_gate_factory_hpp(nb::module_& mgate) {
    mgate.def("I", &gate::I, "Generate general Gate class instance of I.");
    mgate.def("GlobalPhase",
              &gate::GlobalPhase,
              "Generate general Gate class instance of GlobalPhase.",
              "phase"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("X",
              &gate::X,
              "Generate general Gate class instance of X.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Y",
              &gate::Y,
              "Generate general Gate class instance of Y.",
              "taget"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Z",
              &gate::Z,
              "Generate general Gate class instance of Z.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("H",
              &gate::H,
              "Generate general Gate class instance of H.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("S",
              &gate::S,
              "Generate general Gate class instance of S.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Sdag",
              &gate::Sdag,
              "Generate general Gate class instance of Sdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("T",
              &gate::T,
              "Generate general Gate class instance of T.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Tdag",
              &gate::Tdag,
              "Generate general Gate class instance of Tdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtX",
              &gate::SqrtX,
              "Generate general Gate class instance of SqrtX.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtXdag",
              &gate::SqrtXdag,
              "Generate general Gate class instance of SqrtXdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtY",
              &gate::SqrtY,
              "Generate general Gate class instance of SqrtY.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtYdag",
              &gate::SqrtYdag,
              "Generate general Gate class instance of SqrtYdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P0",
              &gate::P0,
              "Generate general Gate class instance of P0.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P1",
              &gate::P1,
              "Generate general Gate class instance of P1.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RX",
              &gate::RX,
              "Generate general Gate class instance of RX.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RY",
              &gate::RY,
              "Generate general Gate class instance of RY.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RZ",
              &gate::RZ,
              "Generate general Gate class instance of RZ.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U1",
              &gate::U1,
              "Generate general Gate class instance of U1.",
              "target"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U2",
              &gate::U2,
              "Generate general Gate class instance of U2.",
              "target"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U3",
              &gate::U3,
              "Generate general Gate class instance of U3.",
              "target"_a,
              "theta"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Swap",
              &gate::Swap,
              "Generate general Gate class instance of Swap.",
              "target1"_a,
              "target2"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def(
        "CX",
        &gate::CX,
        "Generate general Gate class instance of CX.\n\n.. note:: CX is a specialization of X.");
    mgate.def("CNot",
              &gate::CX,
              "Generate general Gate class instance of CNot.\n\n.. note:: CNot is an alias of CX.");
    mgate.def(
        "CZ",
        &gate::CZ,
        "Generate general Gate class instance of CZ.\n\n.. note:: CZ is a specialization of Z.");
    mgate.def(
        "CCX",
        &gate::CCX,
        "Generate general Gate class instance of CXX.\n\n.. note:: CX is a specialization of X.");
    mgate.def(
        "CCNot",
        &gate::CCX,
        "Generate general Gate class instance of CCNot.\n\n.. note:: CCNot is an alias of CCX.");
    mgate.def("Toffoli",
              &gate::CCX,
              "Generate general Gate class instance of Toffoli.\n\n.. note:: Toffoli is an alias "
              "of CCX.");
    mgate.def("OneTargetMatrix",
              &gate::OneTargetMatrix,
              "Generate general Gate class instance of OneTargetMatrix.",
              "target"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("TwoTargetMatrix",
              &gate::TwoTargetMatrix,
              "Generate general Gate class instance of TwoTargetMatrix.",
              "target1"_a,
              "target2"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("DenseMatrix",
              &gate::DenseMatrix,
              "Generate general Gate class instance of DenseMatrix. IGate, OneTargetMatrixGate or "
              "TwoTargetMatrixGate correspond to len(target) is created. The case len(target) >= 3 "
              "is currently not supported.",
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "is_unitary"_a = false);
    mgate.def("Pauli",
              &gate::Pauli,
              "Generate general Gate class instance of Pauli.",
              "pauli"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("PauliRotation",
              &gate::PauliRotation,
              "Generate general Gate class instance of PauliRotation.",
              "pauli"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Probablistic",
              &gate::Probablistic,
              "Generate general Gate class instance of Probablistic.",
              "distribution"_a,
              "gate_list"_a);
}
}  // namespace internal
#endif
}  // namespace scaluq
