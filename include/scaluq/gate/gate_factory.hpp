#pragma once

#include "../util/utility.hpp"
#include "gate_matrix.hpp"
#include "gate_pauli.hpp"
#include "gate_probablistic.hpp"
#include "gate_standard.hpp"

namespace scaluq {
namespace internal {
class GateFactory {
public:
    template <GateImpl T, typename... Args>
    static Gate<typename T::Fp, typename T::Sp> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {

template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> I() {
    return internal::GateFactory::create_gate<internal::IGateImpl<Fp, Sp>>();
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> GlobalPhase(Fp phase, const std::vector<std::uint64_t>& control_qubits = {}) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl<Fp, Sp>>(
        internal::vector_to_mask(control_qubits), phase);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> X(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> Y(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::YGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> Z(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> H(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::HGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> S(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> Sdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> T(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> Tdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> SqrtX(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> SqrtXdag(std::uint64_t target,
                             const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> SqrtY(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> SqrtYdag(std::uint64_t target,
                             const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> P0(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P0GateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> P1(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P1GateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> RX(std::uint64_t target,
                       Fp angle,
                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RXGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> RY(std::uint64_t target,
                       Fp angle,
                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RYGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> RZ(std::uint64_t target,
                       Fp angle,
                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RZGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> U1(std::uint64_t target,
                       Fp lambda,
                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U1GateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), lambda);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> U2(std::uint64_t target,
                       Fp phi,
                       Fp lambda,
                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U2GateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), phi, lambda);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> U3(std::uint64_t target,
                       Fp theta,
                       Fp phi,
                       Fp lambda,
                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U3GateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), theta, phi, lambda);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline auto& CNot = CX<Fp, Sp>;
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> CZ(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> CCX(std::uint64_t control1, std::uint64_t control2, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control1, control2}));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline auto& Toffoli = CCX<Fp, Sp>;
template <std::floating_point Fp, ExecutionSpace Sp>
inline auto& CCNot = CCX<Fp, Sp>;
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> Swap(std::uint64_t target1,
                         std::uint64_t target2,
                         const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl<Fp, Sp>>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> Pauli(const PauliOperator<Fp, Sp>& pauli,
                          const std::vector<std::uint64_t>& controls = {}) {
    auto tar = pauli.target_qubit_list();
    return internal::GateFactory::create_gate<internal::PauliGateImpl<Fp, Sp>>(
        internal::vector_to_mask(controls), pauli);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> PauliRotation(const PauliOperator<Fp, Sp>& pauli,
                                  Fp angle,
                                  const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl<Fp, Sp>>(
        internal::vector_to_mask(controls), pauli, angle);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> DenseMatrix(const std::vector<std::uint64_t>& targets,
                                const internal::ComplexMatrix<Fp>& matrix,
                                const std::vector<std::uint64_t>& controls = {},
                                bool is_unitary = false) {
    std::uint64_t nqubits = targets.size();
    std::uint64_t dim = 1ULL << nqubits;
    if (static_cast<std::uint64_t>(matrix.rows()) != dim ||
        static_cast<std::uint64_t>(matrix.cols()) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<std::uint64_t>&, const "
            "internal::ComplexMatrix<Fp>&): "
            "matrix size must be 2^{n_qubits} x 2^{n_qubits}.");
    }
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Fp, Sp>>(
            internal::vector_to_mask(targets),
            internal::vector_to_mask(controls),
            matrix,
            is_unitary);
    }
    internal::ComplexMatrix<Fp> matrix_transformed =
        internal::transform_dense_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Fp, Sp>>(
        internal::vector_to_mask(targets),
        internal::vector_to_mask(controls),
        matrix_transformed,
        is_unitary);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> SparseMatrix(const std::vector<std::uint64_t>& targets,
                                 const internal::SparseComplexMatrix<Fp>& matrix,
                                 const std::vector<std::uint64_t>& controls = {}) {
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Fp, Sp>>(
            internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix);
    }
    internal::SparseComplexMatrix<Fp> matrix_transformed =
        internal::transform_sparse_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Fp, Sp>>(
        internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix_transformed);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline Gate<Fp, Sp> Probablistic(const std::vector<Fp, Sp>& distribution,
                                 const std::vector<Gate<Fp, Sp>>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl<Fp, Sp>>(distribution,
                                                                                      gate_list);
}
}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp, ExecutionSpace Sp>
void bind_gate_gate_factory_hpp(nb::module_& mgate) {
    mgate.def("I", &gate::I<Fp, Sp>, "Generate general Gate class instance of I.");
    mgate.def("GlobalPhase",
              &gate::GlobalPhase<Fp, Sp>,
              "Generate general Gate class instance of GlobalPhase.",
              "phase"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("X",
              &gate::X<Fp, Sp>,
              "Generate general Gate class instance of X.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Y",
              &gate::Y<Fp, Sp>,
              "Generate general Gate class instance of Y.",
              "taget"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Z",
              &gate::Z<Fp, Sp>,
              "Generate general Gate class instance of Z.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("H",
              &gate::H<Fp, Sp>,
              "Generate general Gate class instance of H.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("S",
              &gate::S<Fp, Sp>,
              "Generate general Gate class instance of S.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Sdag",
              &gate::Sdag<Fp, Sp>,
              "Generate general Gate class instance of Sdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("T",
              &gate::T<Fp, Sp>,
              "Generate general Gate class instance of T.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Tdag",
              &gate::Tdag<Fp, Sp>,
              "Generate general Gate class instance of Tdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtX",
              &gate::SqrtX<Fp, Sp>,
              "Generate general Gate class instance of SqrtX.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtXdag",
              &gate::SqrtXdag<Fp, Sp>,
              "Generate general Gate class instance of SqrtXdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtY",
              &gate::SqrtY<Fp, Sp>,
              "Generate general Gate class instance of SqrtY.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtYdag",
              &gate::SqrtYdag<Fp, Sp>,
              "Generate general Gate class instance of SqrtYdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P0",
              &gate::P0<Fp, Sp>,
              "Generate general Gate class instance of P0.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P1",
              &gate::P1<Fp, Sp>,
              "Generate general Gate class instance of P1.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RX",
              &gate::RX<Fp, Sp>,
              "Generate general Gate class instance of RX.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RY",
              &gate::RY<Fp, Sp>,
              "Generate general Gate class instance of RY.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RZ",
              &gate::RZ<Fp, Sp>,
              "Generate general Gate class instance of RZ.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U1",
              &gate::U1<Fp, Sp>,
              "Generate general Gate class instance of U1.",
              "target"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U2",
              &gate::U2<Fp, Sp>,
              "Generate general Gate class instance of U2.",
              "target"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U3",
              &gate::U3<Fp, Sp>,
              "Generate general Gate class instance of U3.",
              "target"_a,
              "theta"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Swap",
              &gate::Swap<Fp, Sp>,
              "Generate general Gate class instance of Swap.",
              "target1"_a,
              "target2"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def(
        "CX",
        &gate::CX<Fp, Sp>,
        "Generate general Gate class instance of CX.\n\n.. note:: CX is a specialization of X.");
    mgate.def("CNot",
              &gate::CX<Fp, Sp>,
              "Generate general Gate class instance of CNot.\n\n.. note:: CNot is an alias of CX.");
    mgate.def(
        "CZ",
        &gate::CZ<Fp, Sp>,
        "Generate general Gate class instance of CZ.\n\n.. note:: CZ is a specialization of Z.");
    mgate.def(
        "CCX",
        &gate::CCX<Fp, Sp>,
        "Generate general Gate class instance of CXX.\n\n.. note:: CX is a specialization of X.");
    mgate.def(
        "CCNot",
        &gate::CCX<Fp, Sp>,
        "Generate general Gate class instance of CCNot.\n\n.. note:: CCNot is an alias of CCX.");
    mgate.def("Toffoli",
              &gate::CCX<Fp, Sp>,
              "Generate general Gate class instance of Toffoli.\n\n.. note:: Toffoli is an alias "
              "of CCX.");
    mgate.def("DenseMatrix",
              &gate::DenseMatrix<Fp>,
              "Generate general Gate class instance of DenseMatrix.",
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "is_unitary"_a = false);
    mgate.def("SparseMatrix",
              &gate::SparseMatrix<Fp>,
              "Generate general Gate class instance of SparseMatrix.",
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Pauli",
              &gate::Pauli<Fp, Sp>,
              "Generate general Gate class instance of Pauli.",
              "pauli"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("PauliRotation",
              &gate::PauliRotation<Fp, Sp>,
              "Generate general Gate class instance of PauliRotation.",
              "pauli"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Probablistic",
              &gate::Probablistic<Fp, Sp>,
              "Generate general Gate class instance of Probablistic.",
              "distribution"_a,
              "gate_list"_a);
}
}  // namespace internal
#endif
}  // namespace scaluq
