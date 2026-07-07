#pragma once

#include "../util/utility.hpp"
#include "gate_matrix.hpp"
#include "gate_measurement.hpp"
#include "gate_pauli.hpp"
#include "gate_probabilistic.hpp"
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
inline Gate<Prec> GlobalPhase(double phase,
                              const std::vector<std::uint64_t>& controls = {},
                              std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl<Prec>>(
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(phase));
}
template <Precision Prec>
inline Gate<Prec> X(std::uint64_t target,
                    const std::vector<std::uint64_t>& controls = {},
                    std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> Y(std::uint64_t target,
                    const std::vector<std::uint64_t>& controls = {},
                    std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::YGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> Z(std::uint64_t target,
                    const std::vector<std::uint64_t>& controls = {},
                    std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::ZGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> H(std::uint64_t target,
                    const std::vector<std::uint64_t>& controls = {},
                    std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::HGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> S(std::uint64_t target,
                    const std::vector<std::uint64_t>& controls = {},
                    std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::SGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> Sdag(std::uint64_t target,
                       const std::vector<std::uint64_t>& controls = {},
                       std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::SdagGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> T(std::uint64_t target,
                    const std::vector<std::uint64_t>& controls = {},
                    std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::TGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> Tdag(std::uint64_t target,
                       const std::vector<std::uint64_t>& controls = {},
                       std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::TdagGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> SqrtX(std::uint64_t target,
                        const std::vector<std::uint64_t>& controls = {},
                        std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> SqrtXdag(std::uint64_t target,
                           const std::vector<std::uint64_t>& controls = {},
                           std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> SqrtY(std::uint64_t target,
                        const std::vector<std::uint64_t>& controls = {},
                        std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> SqrtYdag(std::uint64_t target,
                           const std::vector<std::uint64_t>& controls = {},
                           std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> P0(std::uint64_t target,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::P0GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> P1(std::uint64_t target,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::P1GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> Measurement(std::uint64_t target,
                              std::uint64_t classical_bit,
                              bool reset = false) {
    return internal::GateFactory::create_gate<internal::MeasurementGateImpl<Prec>>(
        internal::vector_to_mask({target}), classical_bit, reset);
}
template <Precision Prec>
inline Gate<Prec> RX(std::uint64_t target,
                     double angle,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::RXGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec>
inline Gate<Prec> RY(std::uint64_t target,
                     double angle,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::RYGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec>
inline Gate<Prec> RZ(std::uint64_t target,
                     double angle,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::RZGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec>
inline Gate<Prec> U1(std::uint64_t target,
                     double lambda,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::U1GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(lambda));
}
template <Precision Prec>
inline Gate<Prec> U2(std::uint64_t target,
                     double phi,
                     double lambda,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::U2GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(phi),
        static_cast<internal::Float<Prec>>(lambda));
}
template <Precision Prec>
inline Gate<Prec> U3(std::uint64_t target,
                     double theta,
                     double phi,
                     double lambda,
                     const std::vector<std::uint64_t>& controls = {},
                     std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::U3GateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        static_cast<internal::Float<Prec>>(theta),
        static_cast<internal::Float<Prec>>(phi),
        static_cast<internal::Float<Prec>>(lambda));
}
template <Precision Prec>
inline Gate<Prec> CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask({control}),
        internal::vector_to_mask({control}, {1}));
}
template <Precision Prec>
inline auto& CNot = CX<Prec>;
template <Precision Prec>
inline Gate<Prec> CZ(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask({control}),
        internal::vector_to_mask({control}, {1}));
}
template <Precision Prec>
inline Gate<Prec> CCX(std::uint64_t control1, std::uint64_t control2, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Prec>>(
        internal::vector_to_mask({target}),
        internal::vector_to_mask({control1, control2}),
        internal::vector_to_mask({control1, control2}, {1, 1}));
}
template <Precision Prec>
inline auto& Toffoli = CCX<Prec>;
template <Precision Prec>
inline auto& CCNot = CCX<Prec>;
template <Precision Prec>
inline Gate<Prec> Swap(std::uint64_t target1,
                       std::uint64_t target2,
                       const std::vector<std::uint64_t>& controls = {},
                       std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::SwapGateImpl<Prec>>(
        internal::vector_to_mask({target1, target2}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values));
}
template <Precision Prec>
inline Gate<Prec> Ecr(std::uint64_t physical_control,
                      std::uint64_t physical_target,
                      const std::vector<std::uint64_t>& controls = {},
                      std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::EcrGateImpl<Prec>>(
        internal::vector_to_mask({physical_target, physical_control}),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        internal::vector_to_mask({physical_control}),
        internal::vector_to_mask({physical_target}));
}
template <Precision Prec>
inline Gate<Prec> Pauli(const PauliOperator<Prec>& pauli,
                        const std::vector<std::uint64_t>& controls = {},
                        std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    auto tar = pauli.target_qubit_list();
    return internal::GateFactory::create_gate<internal::PauliGateImpl<Prec>>(
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        pauli);
}
template <Precision Prec>
inline Gate<Prec> PauliRotation(const PauliOperator<Prec>& pauli,
                                double angle,
                                const std::vector<std::uint64_t>& controls = {},
                                std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl<Prec>>(
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        pauli,
        static_cast<internal::Float<Prec>>(angle));
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec> DenseMatrix(const std::vector<std::uint64_t>& targets,
                              const ComplexMatrix& matrix,
                              const std::vector<std::uint64_t>& controls = {},
                              std::vector<std::uint64_t> control_values = {},
                              bool is_unitary = false) {
    internal::resize_and_check_control_values(controls, control_values);
    std::uint64_t nqubits = targets.size();
    std::uint64_t dim = 1ULL << nqubits;
    if (static_cast<std::uint64_t>(matrix.rows()) != dim ||
        static_cast<std::uint64_t>(matrix.cols()) != dim) {
        throw std::runtime_error(
            "gate::DenseMatrix(const std::vector<std::uint64_t>&, const "
            "ComplexMatrix&): "
            "matrix size must be 2^{n_qubits} x 2^{n_qubits}.");
    }
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Prec, Space>>(
            internal::vector_to_mask(targets),
            internal::vector_to_mask(controls),
            internal::vector_to_mask(controls, control_values),
            matrix,
            is_unitary);
    }
    ComplexMatrix matrix_transformed = internal::transform_dense_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::DenseMatrixGateImpl<Prec, Space>>(
        internal::vector_to_mask(targets),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        matrix_transformed,
        is_unitary);
}
template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec> SparseMatrix(const std::vector<std::uint64_t>& targets,
                               const SparseComplexMatrix& matrix,
                               const std::vector<std::uint64_t>& controls = {},
                               std::vector<std::uint64_t> control_values = {}) {
    internal::resize_and_check_control_values(controls, control_values);
    if (std::is_sorted(targets.begin(), targets.end())) {
        return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Prec, Space>>(
            internal::vector_to_mask(targets),
            internal::vector_to_mask(controls),
            internal::vector_to_mask(controls, control_values),
            matrix);
    }
    SparseComplexMatrix matrix_transformed =
        internal::transform_sparse_matrix_by_order(matrix, targets);
    return internal::GateFactory::create_gate<internal::SparseMatrixGateImpl<Prec, Space>>(
        internal::vector_to_mask(targets),
        internal::vector_to_mask(controls),
        internal::vector_to_mask(controls, control_values),
        matrix_transformed);
}
template <Precision Prec>
inline Gate<Prec> Probabilistic(const std::vector<double>& distribution,
                                const std::vector<Gate<Prec>>& gate_list) {
    return internal::GateFactory::create_gate<internal::ProbabilisticGateImpl<Prec>>(distribution,
                                                                                     gate_list);
}

template <Precision Prec, ExecutionSpace Space>
inline Gate<Prec> Permutation(const std::vector<std::uint64_t>& dst_indices) {
    const std::uint64_t n_qubits = dst_indices.size();
    if (n_qubits >= sizeof(std::uint64_t) * 8) {
        throw std::runtime_error(
            "gate::Permutation: the size of the qubit system must be less than 64.");
    }

    const std::uint64_t expected_mask = n_qubits == 0 ? 0ULL : ((1ULL << n_qubits) - 1);
    if (internal::vector_to_mask(dst_indices) != expected_mask) {
        throw std::runtime_error(
            "gate::Permutation: dst_indices must be a permutation of 0 to n_qubits - 1.");
    }
    return internal::GateFactory::create_gate<internal::PermutationGateImpl<Prec>>(dst_indices);
}

// corresponding to XGate
template <Precision Prec>
inline Gate<Prec> BitFlipNoise(std::int64_t target, double error_rate) {
    return Probabilistic<Prec>({error_rate, 1 - error_rate}, {X<Prec>(target), I<Prec>()});
}
template <Precision Prec>
inline Gate<Prec> DephasingNoise(std::int64_t target, double error_rate) {
    return Probabilistic<Prec>({error_rate, 1 - error_rate}, {Z<Prec>(target), I<Prec>()});
}
// Y: p*p, X: p(1-p), Z: p(1-p)
template <Precision Prec>
inline Gate<Prec> BitFlipAndDephasingNoise(std::int64_t target, double error_rate) {
    double p0 = error_rate * error_rate;
    double p1 = error_rate * (1 - error_rate);
    double p2 = (1 - error_rate) * (1 - error_rate);
    return Probabilistic<Prec>({p0, p1, p1, p2},
                               {Y<Prec>(target), X<Prec>(target), Z<Prec>(target), I<Prec>()});
}
// X: error_rate/3, Y: error_rate/3, Z: error_rate/3
template <Precision Prec>
inline Gate<Prec> DepolarizingNoise(std::int64_t target, double error_rate) {
    return Probabilistic<Prec>({error_rate / 3, error_rate / 3, error_rate / 3, 1 - error_rate},
                               {X<Prec>(target), Y<Prec>(target), Z<Prec>(target), I<Prec>()});
}

}  // namespace gate

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_factory_hpp(nb::module_& mgate) {
    mgate.def("I",
              &gate::I<Prec>,
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.IGate`.")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "Identity gate instance")
                  .ex(DocString::Code({">>> gate = I()",
                                       ">>> print(gate)",
                                       "Gate Type: I",
                                       "  Target Qubits: {}",
                                       "  Control Qubits: {}",
                                       "  Control Value: {}"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "GlobalPhase",
        &gate::GlobalPhase<Prec>,
        "gamma"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.GlobalPhaseGate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.GlobalPhaseGate` "
                  "class, please downcast it.")
            .arg("gamma", "float", "Global phase angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "Global phase gate instance")
            .ex(DocString::Code({">>> import math",
                                 ">>> gate = GlobalPhase(math.pi/2)",
                                 ">>> print(gate)",
                                 "Gate Type: GlobalPhase",
                                 "  Phase: 1.5708",
                                 "  Target Qubits: {}",
                                 "  Control Qubits: {}",
                                 "  Control Value: {}"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "X",
        &gate::X<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.XGate`. "
                  "Performs bit flip operation.")
            .note("XGate represents the Pauli-X (NOT) gate class.If you need to use functions "
                  "specific to the :class:`~scaluq.XGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "Pauli-X gate instance")
            .ex(DocString::Code({">>> gate = X(0)  # X gate on qubit 0",
                                 ">>> gate = X(1, [0])  # Controlled-X with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "Y",
        &gate::Y<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.YGate`. "
                  "Performs bit flip and phase flip operation.")
            .note("YGate represents the Pauli-Y gate class. If you need to use functions specific "
                  "to the :class:`~scaluq.YGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "Pauli-Y gate instance")
            .ex(DocString::Code({">>> gate = Y(0)  # Y gate on qubit 0",
                                 ">>> gate = Y(1, [0])  # Controlled-Y with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "Z",
        &gate::Z<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.ZGate`. "
                  "Performs bit flip and phase flip operation.")
            .note("ZGate represents the Pauli-Z gate class. If you need to use functions specific "
                  "to the :class:`~scaluq.ZGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "Pauli-Z gate instance")
            .ex(DocString::Code({">>> gate = Z(0)  # Z gate on qubit 0",
                                 ">>> gate = Z(1, [0])  # Controlled-Z with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "H",
        &gate::H<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.HGate`. "
                  "Performs superposition operation.")
            .note("If you need to use functions specific to the :class:`~scaluq.HGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "Hadamard gate instance")
            .ex(DocString::Code({">>> gate = H(0)  # H gate on qubit 0",
                                 ">>> gate = H(1, [0])  # Controlled-H with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "S",
        &gate::S<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.SGate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.SGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "S gate instance")
            .ex(DocString::Code({">>> gate = S(0)  # S gate on qubit 0",
                                 ">>> gate = S(1, [0])  # Controlled-S with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def("Sdag",
              &gate::Sdag<Prec>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.SdagGate`.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.SdagGate` class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "Sdag gate instance")
                  .ex(DocString::Code(
                      {">>> gate = Sdag(0)  # Sdag gate on qubit 0",
                       ">>> gate = Sdag(1, [0])  # Controlled-Sdag with control on qubit 0"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "T",
        &gate::T<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.TGate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.TGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "T gate instance")
            .ex(DocString::Code({">>> gate = T(0)  # T gate on qubit 0",
                                 ">>> gate = T(1, [0])  # Controlled-T with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def("Tdag",
              &gate::Tdag<Prec>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.TdagGate`.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.TdagGate` class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "Tdag gate instance")
                  .ex(DocString::Code(
                      {">>> gate = Tdag(0)  # Tdag gate on qubit 0",
                       ">>> gate = Tdag(1, [0])  # Controlled-Tdag with control on qubit 0"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("SqrtX",
              &gate::SqrtX<Prec>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.SqrtXGate`, represented as "
                        "$\\frac{1}{2}\\begin{bmatrix} 1+i & 1-i \\\\ 1-i & 1+i \\end{bmatrix}$.")
                  .note("If you need to use functions specific to the :class:`~scaluq.SqrtXGate` "
                        "class, please downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "SqrtX gate instance")
                  .ex(DocString::Code({">>> gate = SqrtX(0)  # SqrtX gate on qubit 0",
                                       ">>> gate = SqrtX(1, [0])  # Controlled-SqrtX"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "SqrtXdag",
        &gate::SqrtXdag<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.SqrtXdagGate`, represented as "
                  "$\\begin{bmatrix} 1-i & 1+i\\\\ 1+i "
                  "& 1-i \\end{bmatrix}$.")
            .note("If you need to use functions specific to the :class:`~scaluq.SqrtXdagGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "SqrtXdag gate instance")
            .ex(DocString::Code({">>> gate = SqrtXdag(0)  # SqrtXdag gate on qubit 0",
                                 ">>> gate = SqrtXdag(1, [0])  # Controlled-SqrtXdag"}))
            .build_as_google_style()
            .c_str());
    mgate.def("SqrtY",
              &gate::SqrtY<Prec>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.SqrtYGate`, represented as "
                        "$\\begin{bmatrix} 1+i & -1-i "
                        "\\\\ 1+i & 1+i \\end{bmatrix}$.")
                  .note("If you need to use functions specific to the :class:`~scaluq.SqrtYGate` "
                        "class, please downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "SqrtY gate instance")
                  .ex(DocString::Code({">>> gate = SqrtY(0)  # SqrtY gate on qubit 0",
                                       ">>> gate = SqrtY(1, [0])  # Controlled-SqrtY"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "SqrtYdag",
        &gate::SqrtYdag<Prec>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.SqrtYdagGate`, represented as "
                  "$\\begin{bmatrix} 1-i & 1-i "
                  "\\\\ -1+i & 1-i \\end{bmatrix}$.")
            .note("If you need to use functions specific to the :class:`~scaluq.SqrtYdagGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "SqrtYdag gate instance")
            .ex(DocString::Code({">>> gate = SqrtYdag(0)  # SqrtYdag gate on qubit 0",
                                 ">>> gate = SqrtYdag(1, [0])  # Controlled-SqrtYdag"}))
            .build_as_google_style()
            .c_str());
    mgate.def("P0",
              &gate::P0<Prec>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.P0Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.P0Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "P0 gate instance")
                  .ex(DocString::Code({">>> gate = P0(0)  # P0 gate on qubit 0",
                                       ">>> gate = P0(1, [0])  # Controlled-P0"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("P1",
              &gate::P1<Prec>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.P1Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.P1Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "P1 gate instance")
                  .ex(DocString::Code({">>> gate = P1(0)  # P1 gate on qubit 0",
                                       ">>> gate = P1(1, [0])  # Controlled-P1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "Measurement",
        &gate::Measurement<Prec>,
        "target"_a,
        "classical_bit"_a,
        "reset"_a = false,
        DocString()
            .desc("Generate computational-basis measurement gate.")
            .note("Applying this gate requires a classical register whose size is greater than "
                  "`classical_bit`. If `reset` is true, the target qubit is reset to |0> after "
                  "the measurement.")
            .arg("target", "int", "Target qubit index")
            .arg("classical_bit", "int", "Destination classical bit index")
            .arg("reset", "bool", true, "Whether to reset the target qubit to |0>")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "Measurement gate instance")
            .ex(DocString::Code({">>> gate = Measurement(0, 1)"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "RX",
        &gate::RX<Prec>,
        "target"_a,
        "theta"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate rotation gate around X-axis. Rotation angle is specified in radians.")
            .note("If you need to use functions specific to the :class:`~scaluq.RXGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "RX gate instance")
            .ex(DocString::Code({">>> import math",
                                 ">>> gate = RX(0, math.pi/2)  # π/2 rotation around X-axis",
                                 ">>> gate = RX(1, math.pi, [0])  # Controlled-RX"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "RY",
        &gate::RY<Prec>,
        "target"_a,
        "theta"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate rotation gate around Y-axis. Rotation angle is specified in radians.")
            .note("If you need to use functions specific to the :class:`~scaluq.RYGate` class, "
                  "please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "RY gate instance")
            .ex(DocString::Code({">>> import math",
                                 ">>> gate = RY(0, math.pi/2)  # π/2 rotation around Y-axis",
                                 ">>> gate = RY(1, math.pi, [0])  # Controlled-RY"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "RZ",
        &gate::RZ<Prec>,
        "target"_a,
        "theta"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate rotation gate around Z-axis. Rotation angle is specified in radians.")
            .note("If you need to use functions specific to the :class:`~scaluq.RZGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "RZ gate instance")
            .ex(DocString::Code({">>> import math",
                                 ">>> gate = RZ(0, math.pi/2)  # π/2 rotation around Z-axis",
                                 ">>> gate = RZ(1, math.pi, [0])  # Controlled-RZ"}))
            .build_as_google_style()
            .c_str());
    mgate.def("U1",
              &gate::U1<Prec>,
              "target"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.U1Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.U1Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("lambda_", "float", "Rotation angle in radians")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "U1 gate instance")
                  .ex(DocString::Code({">>> import math",
                                       ">>> gate = U1(0, math.pi/2)  # π/2 rotation around Z-axis",
                                       ">>> gate = U1(1, math.pi, [0])  # Controlled-U1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("U2",
              &gate::U2<Prec>,
              "target"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.U2Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.U2Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("phi", "float", "Rotation angle in radians")
                  .arg("lambda_", "float", "Rotation angle in radians")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "U2 gate instance")
                  .ex(DocString::Code(
                      {">>> import math",
                       ">>> gate = U2(0, math.pi/2, math.pi)  # π/2 rotation around Z-axis",
                       ">>> gate = U2(1, math.pi, math.pi/2, [0])  # Controlled-U2"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "U3",
        &gate::U3<Prec>,
        "target"_a,
        "theta"_a,
        "phi"_a,
        "lambda_"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.U3Gate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.U3Gate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("phi", "float", "Rotation angle in radians")
            .arg("lambda_", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "U3 gate instance")
            .ex(DocString::Code(
                {">>> import math",
                 ">>> gate = U3(0, math.pi/2, math.pi, math.pi)  # π/2 rotation around Z-axis",
                 ">>> gate = U3(1, math.pi, math.pi/2, math.pi, [0])  # Controlled-U3"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "Swap",
        &gate::Swap<Prec>,
        "target1"_a,
        "target2"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate SWAP gate. Swaps the states of two qubits.")
            .note("If you need to use functions specific to the :class:`~scaluq.SwapGate` class, "
                  "please downcast it.")
            .arg("target1", "int", "First target qubit index")
            .arg("target2", "int", "Second target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "SWAP gate instance")
            .ex(DocString::Code({">>> gate = Swap(0, 1)  # Swap qubits 0 and 1",
                                 ">>> gate = Swap(1, 2, [0])  # Controlled-SWAP"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "Ecr",
        &gate::Ecr<Prec>,
        "physical_control"_a,
        "physical_target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate ECR gate. Echoed cross-resonance gate.")
            .note("If you need to use functions specific to the :class:`~scaluq.EcrGate` class, "
                  "please downcast it.")
            .arg("physical_control", "int", "Physical control qubit index")
            .arg("physical_target", "int", "Physical target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "Ecr gate instance")
            .ex(DocString::Code({">>> gate = Ecr(0, 1)  # control : 0 and target : 1",
                                 ">>> gate = Ecr(1, 2, [0])  #Controlled-ECR"}))
            .build_as_google_style()
            .c_str());
    mgate.def("CX",
              &gate::CX<Prec>,
              "control"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.XGate` with one control qubit. Performs "
                        "controlled-X operation.")
                  .note("CX is a specialization of X. If you need to use functions specific to the "
                        ":class:`~scaluq.XGate` class, please downcast it.")
                  .arg("control", "int", "Control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "CX gate instance")
                  .ex(DocString::Code({">>> gate = CX(0, 1)  # CX gate with control on qubit 0",
                                       ">>> gate = CX(1, 2)  # CX gate with control on qubit 1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "CNot",
        &gate::CX<Prec>,
        "control"_a,
        "target"_a,
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.XGate` with "
                  "one control qubit. Performs controlled-X operation.")
            .note("CNot is an alias of CX. If you need to use functions specific to the "
                  ":class:`~scaluq.XGate` class, please downcast it.")
            .arg("control", "int", "Control qubit index")
            .arg("target", "int", "Target qubit index")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .ret("Gate", "CNot gate instance")
            .ex(DocString::Code({">>> gate = CNot(0, 1)  # CNot gate with control on qubit 0",
                                 ">>> gate = CNot(1, 2)  # CNot gate with control on qubit 1"}))
            .build_as_google_style()
            .c_str());
    mgate.def("CZ",
              &gate::CZ<Prec>,
              "control"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.ZGate` with one control qubit. Performs "
                        "controlled-Z operation.")
                  .note("CZ is a specialization of Z. If you need to use functions specific to the "
                        ":class:`~scaluq.ZGate` class, please downcast it.")
                  .arg("control", "int", "Control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "CZ gate instance")
                  .ex(DocString::Code({">>> gate = CZ(0, 1)  # CZ gate with control on qubit 0",
                                       ">>> gate = CZ(1, 2)  # CZ gate with control on qubit 1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("CCX",
              &gate::CCX<Prec>,
              "control1"_a,
              "control2"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.XGate` with two control qubits. Performs "
                        "controlled-controlled-X operation.")
                  .note("If you need to use functions specific to the :class:`~scaluq.XGate` "
                        "class, please downcast it.")
                  .arg("control1", "int", "First control qubit index")
                  .arg("control2", "int", "Second control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "CCX gate instance")
                  .ex(DocString::Code(
                      {">>> gate = CCX(0, 1, 2)  # CCX gate with controls on qubits 0 and 1",
                       ">>> gate = CCX(1, 2, 3)  # CCX gate with controls on qubits 1 and 2"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("CCNot",
              &gate::CCX<Prec>,
              "control1"_a,
              "control2"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.XGate` with two control qubits. Performs "
                        "controlled-controlled-X operation.")
                  .note("CCNot is an alias of CCX. If you need to use functions specific to the "
                        ":class:`~scaluq.XGate` class, please downcast it.")
                  .arg("control1", "int", "First control qubit index")
                  .arg("control2", "int", "Second control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "CCNot gate instance")
                  .ex(DocString::Code(
                      {">>> gate = CCNot(0, 1, 2)  # CCNot gate with controls on qubits 0 and 1",
                       ">>> gate = CCNot(1, 2, 3)  # CCNot gate with controls on qubits 1 and 2"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("Toffoli",
              &gate::CCX<Prec>,
              "control1"_a,
              "control2"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.XGate` with two control qubits. "
                        "Performs controlled-controlled-X operation.")
                  .note("Toffoli is an alias of CCX. If you need to use functions specific to the "
                        ":class:`~scaluq.XGate` class, please downcast it.")
                  .arg("control1", "int", "First control qubit index")
                  .arg("control2", "int", "Second control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "Toffoli gate instance")
                  .ex(DocString::Code({">>> gate = Toffoli(0, 1, 2)  # Toffoli gate with "
                                       "controls on qubits 0 and 1",
                                       ">>> gate = Toffoli(1, 2, 3)  # Toffoli gate with "
                                       "controls on qubits 1 and 2"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("Pauli",
              &gate::Pauli<Prec>,
              "pauli"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.PauliGate`. Performs Pauli operation.")
                  .note("If you need to use functions specific to the :class:`~scaluq.PauliGate` "
                        "class, please downcast it.")
                  .arg("pauli", "PauliOperator", "Pauli operator")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "Pauli gate instance")
                  .ex(DocString::Code({">>> pauli = PauliOperator.from_pauli_string('X 0')",
                                       ">>> gate = Pauli(pauli)",
                                       ">>> gate = Pauli(pauli, [1])  # Controlled-Pauli"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("PauliRotation",
              &gate::PauliRotation<Prec>,
              "pauli"_a,
              "theta"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.PauliRotationGate`. Performs Pauli rotation operation.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.PauliRotationGate` "
                        "class, please downcast it.")
                  .arg("pauli", "PauliOperator", "Pauli operator")
                  .arg("theta", "float", "Rotation angle in radians")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "PauliRotation gate instance")
                  .ex(DocString::Code(
                      {">>> pauli = PauliOperator.from_pauli_string('X 0')",
                       ">>> import math",
                       ">>> gate = PauliRotation(pauli, math.pi/2)",
                       ">>> gate = PauliRotation(pauli, math.pi/2, [1])  # Controlled-Pauli"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("Probabilistic",
              &gate::Probabilistic<Prec>,
              "distribution"_a,
              "gate_list"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.ProbabilisticGate`. Performs probabilistic operation.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.ProbabilisticGate` "
                        "class, please downcast it.")
                  .arg("distribution", "list[float]", "Probabilistic distribution")
                  .arg("gate_list", "list[Gate]", "List of gates")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .ret("Gate", "Probabilistic gate instance")
                  .ex(DocString::Code(
                      {">>> distribution = [0.3, 0.7]",
                       ">>> gate_list = [X(0), Y(0)]",
                       ">>> # X is applied with probability 0.3, Y is applied with probability 0.7",
                       ">>> gate = Probabilistic(distribution, gate_list)"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("BitFlipNoise",
              &gate::BitFlipNoise<Prec>,
              "target"_a,
              "error_rate"_a,
              DocString()
                  .desc("Generates a general Gate class instance of BitFlipNoise. `error_rate` is "
                        "the probability of a bit-flip noise, corresponding to the X gate.")
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "DephasingNoise",
        &gate::DephasingNoise<Prec>,
        "target"_a,
        "error_rate"_a,
        DocString()
            .desc("Generates a general Gate class instance of DephasingNoise. `error_rate` is the "
                  "probability of a dephasing noise, corresponding to the Z gate.")
            .build_as_google_style()
            .c_str());
    mgate.def(
        "BitFlipAndDephasingNoise",
        &gate::BitFlipAndDephasingNoise<Prec>,
        "target"_a,
        "error_rate"_a,
        DocString()
            .desc("Generates a general Gate class instance of BitFlipAndDephasingNoise. "
                  "`error_rate` is the "
                  "probability of both bit-flip noise and dephasing noise, corresponding to the X "
                  "gate and "
                  "Z gate.")
            .build_as_google_style()
            .c_str());
    mgate.def(
        "DepolarizingNoise",
        &gate::DepolarizingNoise<Prec>,
        "target"_a,
        "error_rate"_a,
        DocString()
            .desc(
                "Generates a general Gate class instance of DepolarizingNoise. `error_rate` is the "
                "total probability of depolarizing noise, where an X, Y, or Z gate is applied with "
                "a "
                "probability of `error_rate / 3` each.")
            .build_as_google_style()
            .c_str());
}

template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_factory_hpp(nb::module_& mgate) {
    mgate.def(
        "Permutation",
        &gate::Permutation<Prec, Space>,
        "destination_indices"_a,
        DocString()
            .desc("Generate a gate that permutes qubit positions according to destination indices.")
            .arg("destination_indices",
                 "list[int]",
                 "Packed destination index array where element i is the destination of qubit i")
            .ret("Gate", "Qubit reordering gate instance")
            .ex(DocString::Code({">>> gate = Permutation([2, 0, 3, 1])"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "DenseMatrix",
        &gate::DenseMatrix<Prec, Space>,
        "targets"_a,
        "matrix"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "control_values"_a = std::vector<std::uint64_t>{},
        "is_unitary"_a = false,
        DocString()
            .desc("Generate general :class:`~scaluq.Gate` class instance of "
                  ":class:`~scaluq.DenseMatrixGate`. Performs dense matrix operation.")
            .note("If you need to use functions specific to the :class:`~scaluq.DenseMatrixGate` "
                  "class, please downcast it.")
            .arg("targets", "list[int]", "Target qubit indices")
            .arg("matrix", "numpy.ndarray", "Matrix to be applied")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("control_values", "list[int]", true, "Control qubit values")
            .arg("is_unitary",
                 "bool",
                 true,
                 "Whether the matrix is unitary. When the flag indicating that the gate is "
                 "unitary is set to True, a more efficient implementation is used.")
            .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
            .arg("space", "str", true, "Execution space of the gate. Default is 'default'.")
            .ret("Gate", "DenseMatrix gate instance")
            .ex(DocString::Code(
                {">>> import numpy as np",
                 ">>> matrix = np.array([[1, 0], [0, 1]])",
                 ">>> gate = DenseMatrix([0], matrix)",
                 ">>> gate = DenseMatrix([0], matrix, [1])  # Controlled-DenseMatrix"}))
            .build_as_google_style()
            .c_str());
    mgate.def("SparseMatrix",
              &gate::SparseMatrix<Prec, Space>,
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              "control_values"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.Gate` class instance of "
                        ":class:`~scaluq.SparseMatrixGate`. Performs sparse matrix operation.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.SparseMatrixGate` "
                        "class, please downcast it.")
                  .arg("targets", "list[int]", "Target qubit indices")
                  .arg("matrix", "scipy.sparse.csr_matrix", "Matrix to be applied")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .arg("control_values", "list[int]", true, "Control qubit values")
                  .arg("precision", "str", true, "Precision of the gate. Default is 'f64'.")
                  .arg("space", "str", true, "Execution space of the gate. Default is 'default'.")
                  .ret("Gate", "SparseMatrix gate instance")
                  .ex(DocString::Code(
                      {">>> import scipy",
                       ">>> matrix = scipy.sparse.csr_matrix([[1, 0], [0, 1]])",
                       ">>> gate = SparseMatrix([0], matrix)",
                       ">>> gate = SparseMatrix([0], matrix, [1])  # Controlled-SparseMatrix"}))
                  .build_as_google_style()
                  .c_str());
}
}  // namespace internal
#endif
}  // namespace scaluq
