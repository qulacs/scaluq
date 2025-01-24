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
    static Gate<T::Prec> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {

template <Precision Prec>
inline Gate<Prec> I() {
    return internal::GateFactory::create_gate<internal::IGateImpl<Prec>>();
}
template <Precision Prec>
inline Gate<Prec> GlobalPhase(double phase, const std::vector<std::uint64_t>& control_qubits = {}) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl<Prec>>(
        internal::vector_to_mask(control_qubits), Float<Prec>{phase});
}
template <Precision Prec>
inline Gate<Prec> X(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> Y(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::YGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> Z(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> H(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::HGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> S(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> Sdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> T(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> Tdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> SqrtX(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> SqrtXdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> SqrtY(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> SqrtYdag(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> P0(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P0GateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> P1(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P1GateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> RX(std::uint64_t target,
                     double angle,
                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RXGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::Float<Prec>{angle});
}
template <Precision Prec>
inline Gate<Prec> RY(std::uint64_t target,
                     double angle,
                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RYGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::Float<Prec>{angle});
}
template <Precision Prec>
inline Gate<Prec> RZ(std::uint64_t target,
                     double angle,
                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RZGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::Float<Prec>{angle});
}
template <Precision Prec>
inline Gate<Prec> U1(std::uint64_t target,
                     double lambda,
                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U1GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::Float<Prec>{lambda});
}
template <Precision Prec>
inline Gate<Prec> U2(std::uint64_t target,
                     double phi,
                     double lambda,
                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U2GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::Float<Prec>{phi},
        internal::Float<Prec>{lambda});
}
template <Precision Prec>
inline Gate<Prec> U3(std::uint64_t target,
                     double theta,
                     double phi,
                     double lambda,
                     const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U3GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::Float<Prec>{theta},
        internal::Float<Prec>{phi},
        internal::Float<Prec>{lambda});
}
template <Precision Prec>
inline Gate<Prec> CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <Precision Prec>
inline auto& CNot = CX<Prec>;
template <Precision Prec>
inline Gate<Prec> CZ(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <Precision Prec>
inline Gate<Prec> CCX(std::uint64_t control1, std::uint64_t control2, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control1, control2}));
}
template <Precision Prec>
inline auto& Toffoli = CCX<Prec>;
template <Precision Prec>
inline auto& CCNot = CCX<Prec>;
template <Precision Prec>
inline Gate<Prec> Swap(std::uint64_t target1,
                       std::uint64_t target2,
                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl<Prec>>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls));
}
template <Precision Prec>
inline Gate<Prec> Pauli(const PauliOperator<Prec>& pauli,
                        const std::vector<std::uint64_t>& controls = {}) {
    auto tar = pauli.target_qubit_list();
    return internal::GateFactory::create_gate<internal::PauliGateImpl<Prec>>(
        internal::vector_to_mask(controls), pauli);
}
template <Precision Prec>
inline Gate<Prec> PauliRotation(const PauliOperator<Prec>& pauli,
                                double angle,
                                const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl<Prec>>(
        internal::vector_to_mask(controls), pauli, internal::Float<Prec>{Pangle});
}
template <Precision Prec>
inline Gate<Prec> DenseMatrix(const std::vector<std::uint64_t>& targets,
                              const internal::ComplexMatrix& matrix,
                              const std::vector<std::uint64_t>& controls = {},
                              bool is_unitary = false) {
    std::uint64_t nqubits = targets.size();
    std::uint64_t dim = 1ULL << nqubits;
    if (static_cast<std::uint64_t>(matrix.rows()) != dim ||
        static_cast<std::uint64_t>(matrix.cols()) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<std::uint64_t>&, const "
            "internal::ComplexMatrix&): "
            "matrix size must be 2^{n_qubits} x 2^{n_qubits}.");
    }
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Prec>>(
            internal::vector_to_mask(targets),
            internal::vector_to_mask(controls),
            matrix,
            is_unitary);
    }
    internal::ComplexMatrix matrix_transformed =
        internal::transform_dense_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Prec>>(
        internal::vector_to_mask(targets),
        internal::vector_to_mask(controls),
        matrix_transformed,
        is_unitary);
}
template <Precision Prec>
inline Gate<Prec> SparseMatrix(const std::vector<std::uint64_t>& targets,
                               const internal::SparseComplexMatrix& matrix,
                               const std::vector<std::uint64_t>& controls = {}) {
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Prec>>(
            internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix);
    }
    internal::SparseComplexMatrix<Prec> matrix_transformed =
        internal::transform_sparse_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Prec>>(
        internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix_transformed);
}
template <Precision Prec>
inline Gate<Prec> Probablistic(const std::vector<double>& distribution,
                               const std::vector<Gate<Prec>>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl<Prec>>(distribution,
                                                                                    gate_list);
}
}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_factory_hpp(nb::module_& mgate) {
    mgate.def("I", &gate::I<Prec>, "Generate general Gate class instance of I.");
    mgate.def("GlobalPhase",
              &gate::GlobalPhase<Prec>,
              "Generate general Gate class instance of GlobalPhase.",
              "phase"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("X",
              &gate::X<Prec>,
              "Generate general Gate class instance of X.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Y",
              &gate::Y<Prec>,
              "Generate general Gate class instance of Y.",
              "taget"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Z",
              &gate::Z<Prec>,
              "Generate general Gate class instance of Z.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("H",
              &gate::H<Prec>,
              "Generate general Gate class instance of H.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("S",
              &gate::S<Prec>,
              "Generate general Gate class instance of S.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Sdag",
              &gate::Sdag<Prec>,
              "Generate general Gate class instance of Sdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("T",
              &gate::T<Prec>,
              "Generate general Gate class instance of T.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Tdag",
              &gate::Tdag<Prec>,
              "Generate general Gate class instance of Tdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtX",
              &gate::SqrtX<Prec>,
              "Generate general Gate class instance of SqrtX.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtXdag",
              &gate::SqrtXdag<Prec>,
              "Generate general Gate class instance of SqrtXdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtY",
              &gate::SqrtY<Prec>,
              "Generate general Gate class instance of SqrtY.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtYdag",
              &gate::SqrtYdag<Prec>,
              "Generate general Gate class instance of SqrtYdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P0",
              &gate::P0<Prec>,
              "Generate general Gate class instance of P0.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P1",
              &gate::P1<Prec>,
              "Generate general Gate class instance of P1.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RX",
              &gate::RX<Prec>,
              "Generate general Gate class instance of RX.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RY",
              &gate::RY<Prec>,
              "Generate general Gate class instance of RY.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RZ",
              &gate::RZ<Prec>,
              "Generate general Gate class instance of RZ.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U1",
              &gate::U1<Prec>,
              "Generate general Gate class instance of U1.",
              "target"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U2",
              &gate::U2<Prec>,
              "Generate general Gate class instance of U2.",
              "target"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U3",
              &gate::U3<Prec>,
              "Generate general Gate class instance of U3.",
              "target"_a,
              "theta"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Swap",
              &gate::Swap<Prec>,
              "Generate general Gate class instance of Swap.",
              "target1"_a,
              "target2"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def(
        "CX",
        &gate::CX<Prec>,
        "Generate general Gate class instance of CX.\n\n.. note:: CX is a specialization of X.");
    mgate.def("CNot",
              &gate::CX<Prec>,
              "Generate general Gate class instance of CNot.\n\n.. note:: CNot is an alias of CX.");
    mgate.def(
        "CZ",
        &gate::CZ<Prec>,
        "Generate general Gate class instance of CZ.\n\n.. note:: CZ is a specialization of Z.");
    mgate.def(
        "CCX",
        &gate::CCX<Prec>,
        "Generate general Gate class instance of CXX.\n\n.. note:: CX is a specialization of X.");
    mgate.def(
        "CCNot",
        &gate::CCX<Prec>,
        "Generate general Gate class instance of CCNot.\n\n.. note:: CCNot is an alias of CCX.");
    mgate.def("Toffoli",
              &gate::CCX<Prec>,
              "Generate general Gate class instance of Toffoli.\n\n.. note:: Toffoli is an alias "
              "of CCX.");
    mgate.def("DenseMatrix",
              &gate::DenseMatrix<Prec>,
              "Generate general Gate class instance of DenseMatrix.",
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "is_unitary"_a = false);
    mgate.def("SparseMatrix",
              &gate::SparseMatrix<Prec>,
              "Generate general Gate class instance of SparseMatrix.",
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Pauli",
              &gate::Pauli<Prec>,
              "Generate general Gate class instance of Pauli.",
              "pauli"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("PauliRotation",
              &gate::PauliRotation<Prec>,
              "Generate general Gate class instance of PauliRotation.",
              "pauli"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Probablistic",
              &gate::Probablistic<Prec>,
              "Generate general Gate class instance of Probablistic.",
              "distribution"_a,
              "gate_list"_a);
}
}  // namespace internal
#endif
}  // namespace scaluq
