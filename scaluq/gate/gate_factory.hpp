#pragma once

#include "gate_matrix.hpp"
#include "gate_pauli.hpp"
#include "gate_probablistic.hpp"
#include "gate_standard.hpp"

namespace scaluq {
namespace internal {
class GateFactory {
public:
    template <std::floating_point FloatType, GateImpl T, typename... Args>
    static Gate<FloatType> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {

template <std::floating_point FloatType>
inline Gate<FloatType> I() {
    return internal::GateFactory::create_gate<FloatType, internal::IGateImpl<FloatType>>();
}
template <std::floating_point FloatType>
inline Gate<FloatType> GlobalPhase(FloatType phase,
                                   const std::vector<std::uint64_t>& control_qubits = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::GlobalPhaseGateImpl<FloatType>>(
        internal::vector_to_mask(control_qubits), phase);
}
template <std::floating_point FloatType>
inline Gate<FloatType> X(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::XGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> Y(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::YGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> Z(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::ZGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> H(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::HGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> S(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::SGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> Sdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::SdagGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> T(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::TGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> Tdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::TdagGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> SqrtX(std::uint64_t target,
                             const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::SqrtXGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> SqrtXdag(std::uint64_t target,
                                const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::SqrtXdagGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> SqrtY(std::uint64_t target,
                             const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::SqrtYGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> SqrtYdag(std::uint64_t target,
                                const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::SqrtYdagGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> P0(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::P0GateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> P1(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::P1GateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> RX(std::uint64_t target,
                          FloatType angle,
                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::RXGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point FloatType>
inline Gate<FloatType> RY(std::uint64_t target,
                          FloatType angle,
                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::RYGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point FloatType>
inline Gate<FloatType> RZ(std::uint64_t target,
                          FloatType angle,
                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::RZGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point FloatType>
inline Gate<FloatType> U1(std::uint64_t target,
                          FloatType lambda,
                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::U1GateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), lambda);
}
template <std::floating_point FloatType>
inline Gate<FloatType> U2(std::uint64_t target,
                          FloatType phi,
                          FloatType lambda,
                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::U2GateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), phi, lambda);
}
template <std::floating_point FloatType>
inline Gate<FloatType> U3(std::uint64_t target,
                          FloatType theta,
                          FloatType phi,
                          FloatType lambda,
                          const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::U3GateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), theta, phi, lambda);
}
template <std::floating_point FloatType>
inline Gate<FloatType> OneTargetMatrix(std::uint64_t target,
                                       const std::array<std::array<Complex, 2>, 2>& matrix,
                                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType,
                                              internal::OneTargetMatrixGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), matrix);
}
template <std::floating_point FloatType>
inline Gate<FloatType> CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<FloatType, internal::XGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <std::floating_point FloatType>
inline auto& CNot = CX;
template <std::floating_point FloatType>
inline Gate<FloatType> CZ(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<FloatType, internal::ZGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <std::floating_point FloatType>
inline Gate<FloatType> CCX(std::uint64_t control1, std::uint64_t control2, std::uint64_t target) {
    return internal::GateFactory::create_gate<FloatType, internal::XGateImpl<FloatType>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control1, control2}));
}
template <std::floating_point FloatType>
inline auto& Toffoli = CCX<FloatType>;
template <std::floating_point FloatType>
inline auto& CCNot = CCX<FloatType>;
template <std::floating_point FloatType>
inline Gate<FloatType> Swap(std::uint64_t target1,
                            std::uint64_t target2,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType, internal::SwapGateImpl<FloatType>>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls));
}
template <std::floating_point FloatType>
inline Gate<FloatType> TwoTargetMatrix(std::uint64_t target1,
                                       std::uint64_t target2,
                                       const std::array<std::array<Complex, 4>, 4>& matrix,
                                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType,
                                              internal::TwoTargetMatrixGateImpl<FloatType>>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls), matrix);
}
template <std::floating_point FloatType>
inline Gate<FloatType> Pauli(const PauliOperator<FloatType>& pauli,
                             const std::vector<std::uint64_t>& controls = {}) {
    auto tar = pauli.target_qubit_list();
    return internal::GateFactory::create_gate<FloatType, internal::PauliGateImpl<FloatType>>(
        internal::vector_to_mask(controls), pauli);
}
template <std::floating_point FloatType>
inline Gate<FloatType> PauliRotation(const PauliOperator<FloatType>& pauli,
                                     FloatType angle,
                                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<FloatType,
                                              internal::PauliRotationGateImpl<FloatType>>(
        internal::vector_to_mask(controls), pauli, angle);
}
template <std::floating_point FloatType>
inline Gate<FloatType> DenseMatrix(const std::vector<std::uint64_t>& targets,
                                   const internal::ComplexMatrix& matrix,
                                   const std::vector<std::uint64_t>& controls = {}) {
    std::uint64_t nqubits = targets.size();
    std::uint64_t dim = 1ULL << nqubits;
    if (static_cast<std::uint64_t>(matrix.rows()) != dim ||
        static_cast<std::uint64_t>(matrix.cols()) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<std::uint64_t>&, const internal::ComplexMatrix&): "
            "matrix size must be 2^{n_qubits} x 2^{n_qubits}.");
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
        "gate::DenseMatrix(const std::vector<std::uint64_t>&, const internal::ComplexMatrix&): "
        "DenseMatrix "
        "gate "
        "more "
        "than two qubits is not implemented yet.");
}
template <std::floating_point FloatType>
inline Gate<FloatType> Probablistic(const std::vector<FloatType>& distribution,
                                    const std::vector<Gate<FloatType>>& gate_list) {
    return internal::GateFactory::create_gate<FloatType, internal::ProbablisticGateImpl<FloatType>>(
        distribution, gate_list);
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
              "controls"_a = std::vector<std::uint64_t>{});
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
