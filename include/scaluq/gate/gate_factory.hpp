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
    static Gate<typename T::Fp> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {

template <std::floating_point Fp>
inline Gate<Fp> I() {
    return internal::GateFactory::create_gate<internal::IGateImpl<Fp>>();
}
template <std::floating_point Fp>
inline Gate<Fp> GlobalPhase(Fp phase, const std::vector<std::uint64_t>& control_qubits = {}) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl<Fp>>(
        internal::vector_to_mask(control_qubits), phase);
}
template <std::floating_point Fp>
inline Gate<Fp> X(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> Y(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::YGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> Z(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> H(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::HGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> S(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> Sdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> T(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> Tdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> SqrtX(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> SqrtXdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> SqrtY(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> SqrtYdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> P0(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P0GateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> P1(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P1GateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> RX(std::uint64_t target,
                   Fp angle,
                   const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RXGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point Fp>
inline Gate<Fp> RY(std::uint64_t target,
                   Fp angle,
                   const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RYGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point Fp>
inline Gate<Fp> RZ(std::uint64_t target,
                   Fp angle,
                   const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RZGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), angle);
}
template <std::floating_point Fp>
inline Gate<Fp> U1(std::uint64_t target,
                   Fp lambda,
                   const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U1GateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), lambda);
}
template <std::floating_point Fp>
inline Gate<Fp> U2(std::uint64_t target,
                   Fp phi,
                   Fp lambda,
                   const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U2GateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), phi, lambda);
}
template <std::floating_point Fp>
inline Gate<Fp> U3(std::uint64_t target,
                   Fp theta,
                   Fp phi,
                   Fp lambda,
                   const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U3GateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), theta, phi, lambda);
}
template <std::floating_point Fp>
inline Gate<Fp> OneTargetMatrix(std::uint64_t target,
                                const std::array<std::array<Complex<Fp>, 2>, 2>& matrix,
                                const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::OneTargetMatrixGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls), matrix);
}
template <std::floating_point Fp>
inline Gate<Fp> CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <std::floating_point Fp>
inline auto& CNot = CX;
template <std::floating_point Fp>
inline Gate<Fp> CZ(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <std::floating_point Fp>
inline Gate<Fp> CCX(std::uint64_t control1, std::uint64_t control2, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control1, control2}));
}
template <std::floating_point Fp>
inline auto& Toffoli = CCX<Fp>;
template <std::floating_point Fp>
inline auto& CCNot = CCX<Fp>;
template <std::floating_point Fp>
inline Gate<Fp> Swap(std::uint64_t target1,
                     std::uint64_t target2,
                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl<Fp>>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls));
}
template <std::floating_point Fp>
inline Gate<Fp> TwoTargetMatrix(std::uint64_t target1,
                                std::uint64_t target2,
                                const std::array<std::array<Complex<Fp>, 4>, 4>& matrix,
                                const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TwoTargetMatrixGateImpl<Fp>>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls), matrix);
}
template <std::floating_point Fp>
inline Gate<Fp> Pauli(const PauliOperator<Fp>& pauli,
                      const std::vector<std::uint64_t>& controls = {}) {
    auto tar = pauli.target_qubit_list();
    return internal::GateFactory::create_gate<internal::PauliGateImpl<Fp>>(
        internal::vector_to_mask(controls), pauli);
}
template <std::floating_point Fp>
inline Gate<Fp> PauliRotation(const PauliOperator<Fp>& pauli,
                              Fp angle,
                              const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl<Fp>>(
        internal::vector_to_mask(controls), pauli, angle);
}
template <std::floating_point Fp>
inline Gate<Fp> DenseMatrix(const std::vector<std::uint64_t>& targets,
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
        return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Fp>>(
            internal::vector_to_mask(targets),
            internal::vector_to_mask(controls),
            matrix,
            is_unitary);
    }
    internal::ComplexMatrix<Fp> matrix_transformed =
        internal::transform_dense_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Fp>>(
        internal::vector_to_mask(targets),
        internal::vector_to_mask(controls),
        matrix_transformed,
        is_unitary);
}
template <std::floating_point Fp>
inline Gate<Fp> SparseMatrix(const std::vector<std::uint64_t>& targets,
                             const internal::SparseComplexMatrix<Fp>& matrix,
                             const std::vector<std::uint64_t>& controls = {}) {
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Fp>>(
            internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix);
    }
    internal::SparseComplexMatrix<Fp> matrix_transformed =
        internal::transform_sparse_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Fp>>(
        internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix_transformed);
}
template <std::floating_point Fp>
inline Gate<Fp> Probablistic(const std::vector<Fp>& distribution,
                             const std::vector<Gate<Fp>>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl<Fp>>(distribution,
                                                                                  gate_list);
}
}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_gate_gate_factory_hpp(nb::module_& mgate) {
    mgate.def("I", &gate::I<Fp>, "Generate general Gate class instance of I.");
    mgate.def("GlobalPhase",
              &gate::GlobalPhase<Fp>,
              "Generate general Gate class instance of GlobalPhase.",
              "phase"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("X",
              &gate::X<Fp>,
              "Generate general Gate class instance of X.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Y",
              &gate::Y<Fp>,
              "Generate general Gate class instance of Y.",
              "taget"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Z",
              &gate::Z<Fp>,
              "Generate general Gate class instance of Z.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("H",
              &gate::H<Fp>,
              "Generate general Gate class instance of H.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("S",
              &gate::S<Fp>,
              "Generate general Gate class instance of S.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Sdag",
              &gate::Sdag<Fp>,
              "Generate general Gate class instance of Sdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("T",
              &gate::T<Fp>,
              "Generate general Gate class instance of T.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Tdag",
              &gate::Tdag<Fp>,
              "Generate general Gate class instance of Tdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtX",
              &gate::SqrtX<Fp>,
              "Generate general Gate class instance of SqrtX.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtXdag",
              &gate::SqrtXdag<Fp>,
              "Generate general Gate class instance of SqrtXdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtY",
              &gate::SqrtY<Fp>,
              "Generate general Gate class instance of SqrtY.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtYdag",
              &gate::SqrtYdag<Fp>,
              "Generate general Gate class instance of SqrtYdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P0",
              &gate::P0<Fp>,
              "Generate general Gate class instance of P0.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P1",
              &gate::P1<Fp>,
              "Generate general Gate class instance of P1.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RX",
              &gate::RX<Fp>,
              "Generate general Gate class instance of RX.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RY",
              &gate::RY<Fp>,
              "Generate general Gate class instance of RY.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RZ",
              &gate::RZ<Fp>,
              "Generate general Gate class instance of RZ.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U1",
              &gate::U1<Fp>,
              "Generate general Gate class instance of U1.",
              "target"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U2",
              &gate::U2<Fp>,
              "Generate general Gate class instance of U2.",
              "target"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U3",
              &gate::U3<Fp>,
              "Generate general Gate class instance of U3.",
              "target"_a,
              "theta"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Swap",
              &gate::Swap<Fp>,
              "Generate general Gate class instance of Swap.",
              "target1"_a,
              "target2"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def(
        "CX",
        &gate::CX<Fp>,
        "Generate general Gate class instance of CX.\n\n.. note:: CX is a specialization of X.");
    mgate.def("CNot",
              &gate::CX<Fp>,
              "Generate general Gate class instance of CNot.\n\n.. note:: CNot is an alias of CX.");
    mgate.def(
        "CZ",
        &gate::CZ<Fp>,
        "Generate general Gate class instance of CZ.\n\n.. note:: CZ is a specialization of Z.");
    mgate.def(
        "CCX",
        &gate::CCX<Fp>,
        "Generate general Gate class instance of CXX.\n\n.. note:: CX is a specialization of X.");
    mgate.def(
        "CCNot",
        &gate::CCX<Fp>,
        "Generate general Gate class instance of CCNot.\n\n.. note:: CCNot is an alias of CCX.");
    mgate.def("Toffoli",
              &gate::CCX<Fp>,
              "Generate general Gate class instance of Toffoli.\n\n.. note:: Toffoli is an alias "
              "of CCX.");
    mgate.def("OneTargetMatrix",
              &gate::OneTargetMatrix<Fp>,
              "Generate general Gate class instance of OneTargetMatrix.",
              "target"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("TwoTargetMatrix",
              &gate::TwoTargetMatrix<Fp>,
              "Generate general Gate class instance of TwoTargetMatrix.",
              "target1"_a,
              "target2"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{});
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
              &gate::Pauli<Fp>,
              "Generate general Gate class instance of Pauli.",
              "pauli"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("PauliRotation",
              &gate::PauliRotation<Fp>,
              "Generate general Gate class instance of PauliRotation.",
              "pauli"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Probablistic",
              &gate::Probablistic<Fp>,
              "Generate general Gate class instance of Probablistic.",
              "distribution"_a,
              "gate_list"_a);
}
}  // namespace internal
#endif
}  // namespace scaluq
