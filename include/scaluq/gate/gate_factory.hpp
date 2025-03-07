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
    static Gate<T::Prec, typename T::Space> create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};
}  // namespace internal
namespace gate {

template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> I() {
    return internal::GateFactory::create_gate<internal::IGateImpl<Prec, Space>>();
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> GlobalPhase(double phase,
                                     const std::vector<std::uint64_t>& control_qubits = {}) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl<Prec, Space>>(
        internal::vector_to_mask(control_qubits), static_cast<internal::Float<Prec>>(phase));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> X(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> Y(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::YGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> Z(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> H(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::HGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> S(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> Sdag(std::uint64_t target,
                              const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> T(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> Tdag(std::uint64_t target,
                              const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> SqrtX(std::uint64_t target,
                               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> SqrtXdag(std::uint64_t target,
                                  const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> SqrtY(std::uint64_t target,
                               const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> SqrtYdag(std::uint64_t target,
                                  const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> P0(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P0GateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> P1(std::uint64_t target, const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::P1GateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> RX(std::uint64_t target,
                            double angle,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RXGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> RY(std::uint64_t target,
                            double angle,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RYGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> RZ(std::uint64_t target,
                            double angle,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::RZGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> U1(std::uint64_t target,
                            double lambda,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U1GateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(lambda));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> U2(std::uint64_t target,
                            double phi,
                            double lambda,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U2GateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(phi),
        static_cast<internal::Float<Prec>>(lambda));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> U3(std::uint64_t target,
                            double theta,
                            double phi,
                            double lambda,
                            const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::U3GateImpl<Prec, Space>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        static_cast<internal::Float<Prec>>(theta),
        static_cast<internal::Float<Prec>>(phi),
        static_cast<internal::Float<Prec>>(lambda));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <Precision Prec, ExecutionSpace Space>
inline auto& CNot = CX<Prec>;
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> CZ(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> CCX(std::uint64_t control1, std::uint64_t control2, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec, Space>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control1, control2}));
}
template <Precision Prec, ExecutionSpace Space>
inline auto& Toffoli = CCX<Prec, Space>;
template <Precision Prec, ExecutionSpace Space>
inline auto& CCNot = CCX<Prec, Space>;
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> Swap(std::uint64_t target1,
                              std::uint64_t target2,
                              const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl<Prec, Space>>(
        internal::vector_to_mask({target1, target2}), internal::vector_to_mask(controls));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> Pauli(const PauliOperator<Prec, Space>& pauli,
                               const std::vector<std::uint64_t>& controls = {}) {
    auto tar = pauli.target_qubit_list();
    return internal::GateFactory::create_gate<internal::PauliGateImpl<Prec, Space>>(
        internal::vector_to_mask(controls), pauli);
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> PauliRotation(const PauliOperator<Prec, Space>& pauli,
                                       double angle,
                                       const std::vector<std::uint64_t>& controls = {}) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl<Prec, Space>>(
        internal::vector_to_mask(controls), pauli, static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> DenseMatrix(const std::vector<std::uint64_t>& targets,
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
        return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Prec, Space>>(
            internal::vector_to_mask(targets),
            internal::vector_to_mask(controls),
            matrix,
            is_unitary);
    }
    internal::ComplexMatrix matrix_transformed =
        internal::transform_dense_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Prec, Space>>(
        internal::vector_to_mask(targets),
        internal::vector_to_mask(controls),
        matrix_transformed,
        is_unitary);
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> SparseMatrix(const std::vector<std::uint64_t>& targets,
                                      const internal::SparseComplexMatrix& matrix,
                                      const std::vector<std::uint64_t>& controls = {}) {
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Prec, Space>>(
            internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix);
    }
    internal::SparseComplexMatrix matrix_transformed =
        internal::transform_sparse_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Prec, Space>>(
        internal::vector_to_mask(targets), internal::vector_to_mask(controls), matrix_transformed);
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> Probablistic(const std::vector<double>& distribution,
                                      const std::vector<Gate<Prec, Space>>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbablisticGateImpl<Prec, Space>>(
        distribution, gate_list);
}

// bit-flip error
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> XNoise(std::int64_t target, double error_rate) {
    return Probablistic({error_rate, 1 - error_rate}, {X<Prec, Space>(target), I<Prec, Space>()});
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> YNoise(std::int64_t target, double error_rate) {
    return Probablistic({error_rate, 1 - error_rate}, {Y<Prec, Space>(target), I<Prec, Space>()});
}
// phase-flip error
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> ZNoise(std::int64_t target, double error_rate) {
    return Probablistic({error_rate, 1 - error_rate}, {Z<Prec, Space>(target), I<Prec, Space>()});
}
// Y: p*p, X: p(1-p), Z: p(1-p)
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> IndependentXZNoise(std::int64_t target, double error_rate) {
    double p0 = error_rate * error_rate;
    double p1 = error_rate * (1 - error_rate);
    double p2 = (1 - error_rate) * (1 - error_rate);
    return Probablistic(
        {p0, p1, p1, p2},
        {Y<Prec, Space>(target), X<Prec, Space>(target), Z<Prec, Space>(target), I<Prec, Space>()});
}
// X: error_rate/3, Y: error_rate/3, Z: error_rate/3
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec, Space> XYZNoise(std::int64_t target, double error_rate) {
    return Probablistic(
        {error_rate / 3, error_rate / 3, error_rate / 3, 1 - error_rate},
        {X<Prec, Space>(target), Y<Prec, Space>(target), Z<Prec, Space>(target), I<Prec, Space>()});
}

}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_factory_hpp(nb::module_& mgate) {
    mgate.def("I", &gate::I<Prec, Space>, "Generate general Gate class instance of I.");
    mgate.def("GlobalPhase",
              &gate::GlobalPhase<Prec, Space>,
              "Generate general Gate class instance of GlobalPhase.",
              "phase"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("X",
              &gate::X<Prec, Space>,
              "Generate general Gate class instance of X.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Y",
              &gate::Y<Prec, Space>,
              "Generate general Gate class instance of Y.",
              "taget"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Z",
              &gate::Z<Prec, Space>,
              "Generate general Gate class instance of Z.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("H",
              &gate::H<Prec, Space>,
              "Generate general Gate class instance of H.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("S",
              &gate::S<Prec, Space>,
              "Generate general Gate class instance of S.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Sdag",
              &gate::Sdag<Prec, Space>,
              "Generate general Gate class instance of Sdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("T",
              &gate::T<Prec, Space>,
              "Generate general Gate class instance of T.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Tdag",
              &gate::Tdag<Prec, Space>,
              "Generate general Gate class instance of Tdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtX",
              &gate::SqrtX<Prec, Space>,
              "Generate general Gate class instance of SqrtX.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtXdag",
              &gate::SqrtXdag<Prec, Space>,
              "Generate general Gate class instance of SqrtXdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtY",
              &gate::SqrtY<Prec, Space>,
              "Generate general Gate class instance of SqrtY.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("SqrtYdag",
              &gate::SqrtYdag<Prec, Space>,
              "Generate general Gate class instance of SqrtYdag.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P0",
              &gate::P0<Prec, Space>,
              "Generate general Gate class instance of P0.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("P1",
              &gate::P1<Prec, Space>,
              "Generate general Gate class instance of P1.",
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RX",
              &gate::RX<Prec, Space>,
              "Generate general Gate class instance of RX.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RY",
              &gate::RY<Prec, Space>,
              "Generate general Gate class instance of RY.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("RZ",
              &gate::RZ<Prec, Space>,
              "Generate general Gate class instance of RZ.",
              "target"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U1",
              &gate::U1<Prec, Space>,
              "Generate general Gate class instance of U1.",
              "target"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U2",
              &gate::U2<Prec, Space>,
              "Generate general Gate class instance of U2.",
              "target"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("U3",
              &gate::U3<Prec, Space>,
              "Generate general Gate class instance of U3.",
              "target"_a,
              "theta"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Swap",
              &gate::Swap<Prec, Space>,
              "Generate general Gate class instance of Swap.",
              "target1"_a,
              "target2"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def(
        "CX",
        &gate::CX<Prec, Space>,
        "Generate general Gate class instance of CX.\n\n.. note:: CX is a specialization of X.");
    mgate.def("CNot",
              &gate::CX<Prec, Space>,
              "Generate general Gate class instance of CNot.\n\n.. note:: CNot is an alias of CX.");
    mgate.def(
        "CZ",
        &gate::CZ<Prec, Space>,
        "Generate general Gate class instance of CZ.\n\n.. note:: CZ is a specialization of Z.");
    mgate.def(
        "CCX",
        &gate::CCX<Prec, Space>,
        "Generate general Gate class instance of CXX.\n\n.. note:: CX is a specialization of X.");
    mgate.def(
        "CCNot",
        &gate::CCX<Prec, Space>,
        "Generate general Gate class instance of CCNot.\n\n.. note:: CCNot is an alias of CCX.");
    mgate.def("Toffoli",
              &gate::CCX<Prec, Space>,
              "Generate general Gate class instance of Toffoli.\n\n.. note:: Toffoli is an alias "
              "of CCX.");
    mgate.def("DenseMatrix",
              &gate::DenseMatrix<Prec, Space>,
              "Generate general Gate class instance of DenseMatrix.",
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "is_unitary"_a = false);
    mgate.def("SparseMatrix",
              &gate::SparseMatrix<Prec, Space>,
              "Generate general Gate class instance of SparseMatrix.",
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Pauli",
              &gate::Pauli<Prec, Space>,
              "Generate general Gate class instance of Pauli.",
              "pauli"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("PauliRotation",
              &gate::PauliRotation<Prec, Space>,
              "Generate general Gate class instance of PauliRotation.",
              "pauli"_a,
              "angle"_a,
              "controls"_a = std::vector<std::uint64_t>{});
    mgate.def("Probablistic",
              &gate::Probablistic<Prec, Space>,
              "Generate general Gate class instance of Probablistic.",
              "distribution"_a,
              "gate_list"_a);
}
}  // namespace internal
#endif
}  // namespace scaluq
