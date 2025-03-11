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
inline Gate<Fp> CX(std::uint64_t control, std::uint64_t target) {
    return internal::GateFactory::create_gate<internal::XGateImpl<Fp>>(
        internal::vector_to_mask({target}), internal::vector_to_mask({control}));
}
template <std::floating_point Fp>
inline auto& CNot = CX<Fp>;
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
    mgate.def("I",
              &gate::I<Fp>,
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.IGate`.")
                  .ret("Gate", "Identity gate instance")
                  .ex(DocString::Code({">>> gate = I()", ">>> print(gate)", "Identity Gate"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "GlobalPhase",
        &gate::GlobalPhase<Fp>,
        "gamma"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.GlobalPhaseGate`.")
            .note(
                "If you need to use functions specific to the :class:`~scaluq.f64.GlobalPhaseGate` "
                "class, please downcast it.")
            .arg("gamma", "float", "Global phase angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "Global phase gate instance")
            .ex(DocString::Code(
                {">>> gate = GlobalPhase(math.pi/2)", ">>> print(gate)", "Global Phase Gate"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "X",
        &gate::X<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.XGate`. "
                  "Performs bit flip operation.")
            .note("XGate represents the Pauli-X (NOT) gate class.If you need to use functions "
                  "specific to the :class:`~scaluq.f64.XGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "Pauli-X gate instance")
            .ex(DocString::Code({">>> gate = X(0)  # X gate on qubit 0",
                                 ">>> gate = X(1, [0])  # Controlled-X with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "Y",
        &gate::Y<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.YGate`. "
                  "Performs bit flip and phase flip operation.")
            .note("YGate represents the Pauli-Y gate class. If you need to use functions specific "
                  "to the :class:`~scaluq.f64.YGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "Pauli-Y gate instance")
            .ex(DocString::Code({">>> gate = Y(0)  # Y gate on qubit 0",
                                 ">>> gate = Y(1, [0])  # Controlled-Y with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "Z",
        &gate::Z<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.ZGate`. "
                  "Performs bit flip and phase flip operation.")
            .note("ZGate represents the Pauli-Z gate class. If you need to use functions specific "
                  "to the :class:`~scaluq.f64.ZGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "Pauli-Z gate instance")
            .ex(DocString::Code({">>> gate = Z(0)  # Z gate on qubit 0",
                                 ">>> gate = Z(1, [0])  # Controlled-Z with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "H",
        &gate::H<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.HGate`. "
                  "Performs superposition operation.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.HGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "Hadamard gate instance")
            .ex(DocString::Code({">>> gate = H(0)  # H gate on qubit 0",
                                 ">>> gate = H(1, [0])  # Controlled-H with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "S",
        &gate::S<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.SGate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.SGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "S gate instance")
            .ex(DocString::Code({">>> gate = S(0)  # S gate on qubit 0",
                                 ">>> gate = S(1, [0])  # Controlled-S with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def("Sdag",
              &gate::Sdag<Fp>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.SdagGate`.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.f64.SdagGate` class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "Sdag gate instance")
                  .ex(DocString::Code(
                      {">>> gate = Sdag(0)  # Sdag gate on qubit 0",
                       ">>> gate = Sdag(1, [0])  # Controlled-Sdag with control on qubit 0"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "T",
        &gate::T<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.TGate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.TGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "T gate instance")
            .ex(DocString::Code({">>> gate = T(0)  # T gate on qubit 0",
                                 ">>> gate = T(1, [0])  # Controlled-T with control on qubit 0"}))
            .build_as_google_style()
            .c_str());
    mgate.def("Tdag",
              &gate::Tdag<Fp>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.TdagGate`.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.f64.TdagGate` class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "Tdag gate instance")
                  .ex(DocString::Code(
                      {">>> gate = Tdag(0)  # Tdag gate on qubit 0",
                       ">>> gate = Tdag(1, [0])  # Controlled-Tdag with control on qubit 0"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "SqrtX",
        &gate::SqrtX<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.SqrtXGate`, represented as "
                  "$\\frac{1}{2}\\begin{bmatrix} 1+i & 1-i \\\\ 1-i & 1+i \\end{bmatrix}$.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.SqrtXGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "SqrtX gate instance")
            .ex(DocString::Code({">>> gate = SqrtX(0)  # SqrtX gate on qubit 0",
                                 ">>> gate = SqrtX(1, [0])  # Controlled-SqrtX"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "SqrtXdag",
        &gate::SqrtXdag<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.SqrtXdagGate`, represented as "
                  "$\\begin{bmatrix} 1-i & 1+i\\\\ 1+i "
                  "& 1-i \\end{bmatrix}$.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.SqrtXdagGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "SqrtXdag gate instance")
            .ex(DocString::Code({">>> gate = SqrtXdag(0)  # SqrtXdag gate on qubit 0",
                                 ">>> gate = SqrtXdag(1, [0])  # Controlled-SqrtXdag"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "SqrtY",
        &gate::SqrtY<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.SqrtYGate`, represented as "
                  "$\\begin{bmatrix} 1+i & -1-i "
                  "\\\\ 1+i & 1+i \\end{bmatrix}$.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.SqrtYGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "SqrtY gate instance")
            .ex(DocString::Code({">>> gate = SqrtY(0)  # SqrtY gate on qubit 0",
                                 ">>> gate = SqrtY(1, [0])  # Controlled-SqrtY"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "SqrtYdag",
        &gate::SqrtYdag<Fp>,
        "target"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.SqrtYdagGate`, represented as "
                  "$\\begin{bmatrix} 1-i & 1-i "
                  "\\\\ -1+i & 1-i \\end{bmatrix}$.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.SqrtYdagGate` "
                  "class, please downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "SqrtYdag gate instance")
            .ex(DocString::Code({">>> gate = SqrtYdag(0)  # SqrtYdag gate on qubit 0",
                                 ">>> gate = SqrtYdag(1, [0])  # Controlled-SqrtYdag"}))
            .build_as_google_style()
            .c_str());
    mgate.def("P0",
              &gate::P0<Fp>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.P0Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.f64.P0Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "P0 gate instance")
                  .ex(DocString::Code({">>> gate = P0(0)  # P0 gate on qubit 0",
                                       ">>> gate = P0(1, [0])  # Controlled-P0"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("P1",
              &gate::P1<Fp>,
              "target"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.P1Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.f64.P1Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "P1 gate instance")
                  .ex(DocString::Code({">>> gate = P1(0)  # P1 gate on qubit 0",
                                       ">>> gate = P1(1, [0])  # Controlled-P1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "RX",
        &gate::RX<Fp>,
        "target"_a,
        "theta"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate rotation gate around X-axis. Rotation angle is specified in radians.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.RXGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "RX gate instance")
            .ex(DocString::Code({">>> gate = RX(0, math.pi/2)  # π/2 rotation around X-axis",
                                 ">>> gate = RX(1, math.pi, [0])  # Controlled-RX"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "RY",
        &gate::RY<Fp>,
        "target"_a,
        "theta"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate rotation gate around Y-axis. Rotation angle is specified in radians.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.RYGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "RY gate instance")
            .ex(DocString::Code({">>> gate = RY(0, math.pi/2)  # π/2 rotation around Y-axis",
                                 ">>> gate = RY(1, math.pi, [0])  # Controlled-RY"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "RZ",
        &gate::RZ<Fp>,
        "target"_a,
        "theta"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate rotation gate around Z-axis. Rotation angle is specified in radians.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.RZGate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "RZ gate instance")
            .ex(DocString::Code({">>> gate = RZ(0, math.pi/2)  # π/2 rotation around Z-axis",
                                 ">>> gate = RZ(1, math.pi, [0])  # Controlled-RZ"}))
            .build_as_google_style()
            .c_str());
    mgate.def("U1",
              &gate::U1<Fp>,
              "target"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.U1Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.f64.U1Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("lambda_", "float", "Rotation angle in radians")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "U1 gate instance")
                  .ex(DocString::Code({">>> gate = U1(0, math.pi/2)  # π/2 rotation around Z-axis",
                                       ">>> gate = U1(1, math.pi, [0])  # Controlled-U1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("U2",
              &gate::U2<Fp>,
              "target"_a,
              "phi"_a,
              "lambda_"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.U2Gate`.")
                  .note("If you need to use functions specific to the :class:`~scaluq.f64.U2Gate` "
                        "class, please "
                        "downcast it.")
                  .arg("target", "int", "Target qubit index")
                  .arg("phi", "float", "Rotation angle in radians")
                  .arg("lambda_", "float", "Rotation angle in radians")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "U2 gate instance")
                  .ex(DocString::Code(
                      {">>> gate = U2(0, math.pi/2, math.pi)  # π/2 rotation around Z-axis",
                       ">>> gate = U2(1, math.pi, math.pi/2, [0])  # Controlled-U2"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "U3",
        &gate::U3<Fp>,
        "target"_a,
        "theta"_a,
        "phi"_a,
        "hoge_"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.U3Gate`.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.U3Gate` class, "
                  "please "
                  "downcast it.")
            .arg("target", "int", "Target qubit index")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("phi", "float", "Rotation angle in radians")
            .arg("fuga_", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "U3 gate instance")
            .ex(DocString::Code(
                {">>> gate = U3(0, math.pi/2, math.pi, math.pi)  # π/2 rotation around Z-axis",
                 ">>> gate = U3(1, math.pi, math.pi/2, math.pi, [0])  # Controlled-U3"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "Swap",
        &gate::Swap<Fp>,
        "target1"_a,
        "target2"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate SWAP gate. Swaps the states of two qubits.")
            .note(
                "If you need to use functions specific to the :class:`~scaluq.f64.SwapGate` class, "
                "please downcast it.")
            .arg("target1", "int", "First target qubit index")
            .arg("target2", "int", "Second target qubit index")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "SWAP gate instance")
            .ex(DocString::Code({">>> gate = Swap(0, 1)  # Swap qubits 0 and 1",
                                 ">>> gate = Swap(1, 2, [0])  # Controlled-SWAP"}))
            .build_as_google_style()
            .c_str());
    mgate.def("CX",
              &gate::CX<Fp>,
              "control"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.XGate` with one control qubit. Performs "
                        "controlled-X operation.")
                  .note("CX is a specialization of X. If you need to use functions specific to the "
                        ":class:`~scaluq.f64.XGate` class, please downcast it.")
                  .arg("control", "int", "Control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .ret("Gate", "CX gate instance")
                  .ex(DocString::Code({">>> gate = CX(0, 1)  # CX gate with control on qubit 0",
                                       ">>> gate = CX(1, 2)  # CX gate with control on qubit 1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "CNot",
        &gate::CX<Fp>,
        "control"_a,
        "target"_a,
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.XGate` with "
                  "one control qubit. Performs controlled-X operation.")
            .note("CNot is an alias of CX. If you need to use functions specific to the "
                  ":class:`~scaluq.f64.XGate` class, please downcast it.")
            .arg("control", "int", "Control qubit index")
            .arg("target", "int", "Target qubit index")
            .ret("Gate", "CNot gate instance")
            .ex(DocString::Code({">>> gate = CNot(0, 1)  # CNot gate with control on qubit 0",
                                 ">>> gate = CNot(1, 2)  # CNot gate with control on qubit 1"}))
            .build_as_google_style()
            .c_str());
    mgate.def("CZ",
              &gate::CZ<Fp>,
              "control"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.ZGate` with one control qubit. Performs "
                        "controlled-Z operation.")
                  .note("CZ is a specialization of Z. If you need to use functions specific to the "
                        ":class:`~scaluq.f64.ZGate` class, please downcast it.")
                  .arg("control", "int", "Control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .ret("Gate", "CZ gate instance")
                  .ex(DocString::Code({">>> gate = CZ(0, 1)  # CZ gate with control on qubit 0",
                                       ">>> gate = CZ(1, 2)  # CZ gate with control on qubit 1"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("CCX",
              &gate::CCX<Fp>,
              "control1"_a,
              "control2"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.XGate` with two control qubits. Performs "
                        "controlled-controlled-X operation.")
                  .note("If you need to use functions specific to the :class:`~scaluq.f64.XGate` "
                        "class, please downcast it.")
                  .arg("control1", "int", "First control qubit index")
                  .arg("control2", "int", "Second control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .ret("Gate", "CCX gate instance")
                  .ex(DocString::Code(
                      {">>> gate = CCX(0, 1, 2)  # CCX gate with controls on qubits 0 and 1",
                       ">>> gate = CCX(1, 2, 3)  # CCX gate with controls on qubits 1 and 2"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def("CCNot",
              &gate::CCX<Fp>,
              "control1"_a,
              "control2"_a,
              "target"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.XGate` with two control qubits. Performs "
                        "controlled-controlled-X operation.")
                  .note("CCNot is an alias of CCX. If you need to use functions specific to the "
                        ":class:`~scaluq.f64.XGate` class, please downcast it.")
                  .arg("control1", "int", "First control qubit index")
                  .arg("control2", "int", "Second control qubit index")
                  .arg("target", "int", "Target qubit index")
                  .ret("Gate", "CCNot gate instance")
                  .ex(DocString::Code(
                      {">>> gate = CCNot(0, 1, 2)  # CCNot gate with controls on qubits 0 and 1",
                       ">>> gate = CCNot(1, 2, 3)  # CCNot gate with controls on qubits 1 and 2"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "Toffoli",
        &gate::CCX<Fp>,
        "control1"_a,
        "control2"_a,
        "target"_a,
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.XGate` with two control qubits. "
                  "Performs controlled-controlled-X operation.")
            .note("Toffoli is an alias of CCX. If you need to use functions specific to the "
                  ":class:`~scaluq.f64.XGate` class, please downcast it.")
            .arg("control1", "int", "First control qubit index")
            .arg("control2", "int", "Second control qubit index")
            .arg("target", "int", "Target qubit index")
            .ret("Gate", "Toffoli gate instance")
            .ex(DocString::Code(
                {">>> gate = Toffoli(0, 1, 2)  # Toffoli gate with controls on qubits 0 and 1",
                 ">>> gate = Toffoli(1, 2, 3)  # Toffoli gate with controls on qubits 1 and 2"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "DenseMatrix",
        &gate::DenseMatrix<Fp>,
        "targets"_a,
        "matrix"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        "is_unitary"_a = false,
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.DenseMatrixGate`. Performs dense matrix operation.")
            .note(
                "If you need to use functions specific to the :class:`~scaluq.f64.DenseMatrixGate` "
                "class, please downcast it.")
            .arg("targets", "list[int]", "Target qubit indices")
            .arg("matrix", "numpy.ndarray", "Matrix to be applied")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .arg("is_unitary",
                 "bool",
                 true,
                 "Whether the matrix is unitary. When the flag indicating that the gate is "
                 "unitary is set to True, a more efficient implementation is used.")
            .ret("Gate", "DenseMatrix gate instance")
            .ex(DocString::Code(
                {">>> matrix = np.array([[1, 0], [0, 1]])",
                 ">>> gate = DenseMatrix([0], matrix)",
                 ">>> gate = DenseMatrix([0], matrix, [1])  # Controlled-DenseMatrix"}))
            .build_as_google_style()
            .c_str());
    mgate.def("SparseMatrix",
              &gate::SparseMatrix<Fp>,
              "targets"_a,
              "matrix"_a,
              "controls"_a = std::vector<std::uint64_t>{},
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.SparseMatrixGate`. Performs sparse matrix operation.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.f64.SparseMatrixGate` "
                        "class, please downcast it.")
                  .arg("targets", "list[int]", "Target qubit indices")
                  .arg("matrix", "scipy.sparse.csr_matrix", "Matrix to be applied")
                  .arg("controls", "list[int]", true, "Control qubit indices")
                  .ret("Gate", "SparseMatrix gate instance")
                  .ex(DocString::Code(
                      {">>> matrix = scipy.sparse.csr_matrix([[1, 0], [0, 1]])",
                       ">>> gate = SparseMatrix([0], matrix)",
                       ">>> gate = SparseMatrix([0], matrix, [1])  # Controlled-SparseMatrix"}))
                  .build_as_google_style()
                  .c_str());
    mgate.def(
        "Pauli",
        &gate::Pauli<Fp>,
        "pauli"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.PauliGate`. Performs Pauli operation.")
            .note("If you need to use functions specific to the :class:`~scaluq.f64.PauliGate` "
                  "class, please downcast it.")
            .arg("pauli", "PauliOperator", "Pauli operator")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "Pauli gate instance")
            .ex(DocString::Code({">>> pauli = PauliOperator('X 0')",
                                 ">>> gate = Pauli(pauli)",
                                 ">>> gate = Pauli(pauli, [1])  # Controlled-Pauli"}))
            .build_as_google_style()
            .c_str());
    mgate.def(
        "PauliRotation",
        &gate::PauliRotation<Fp>,
        "pauli"_a,
        "theta"_a,
        "controls"_a = std::vector<std::uint64_t>{},
        DocString()
            .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                  ":class:`~scaluq.f64.PauliRotationGate`. Performs Pauli rotation operation.")
            .note("If you need to use functions specific to the "
                  ":class:`~scaluq.f64.PauliRotationGate` "
                  "class, please downcast it.")
            .arg("pauli", "PauliOperator", "Pauli operator")
            .arg("theta", "float", "Rotation angle in radians")
            .arg("controls", "list[int]", true, "Control qubit indices")
            .ret("Gate", "PauliRotation gate instance")
            .ex(DocString::Code(
                {">>> pauli = PauliOperator('X', 0)",
                 ">>> gate = PauliRotation(pauli, math.pi/2)",
                 ">>> gate = PauliRotation(pauli, math.pi/2, [1])  # Controlled-Pauli"}))
            .build_as_google_style()
            .c_str());
    mgate.def("Probablistic",
              &gate::Probablistic<Fp>,
              "distribution"_a,
              "gate_list"_a,
              DocString()
                  .desc("Generate general :class:`~scaluq.f64.Gate` class instance of "
                        ":class:`~scaluq.f64.ProbablisticGate`. Performs probablistic operation.")
                  .note("If you need to use functions specific to the "
                        ":class:`~scaluq.f64.ProbablisticGate` "
                        "class, please downcast it.")
                  .arg("distribution", "list[float]", "Probablistic distribution")
                  .arg("gate_list", "list[Gate]", "List of gates")
                  .ret("Gate", "Probablistic gate instance")
                  .ex(DocString::Code(
                      {">>> distribution = [0.3, 0.7]",
                       ">>> gate_list = [X(0), Y(0)]",
                       ">>> # X is applied with probability 0.3, Y is applied with probability 0.7",
                       ">>> gate = Probablistic(distribution, gate_list)"}))
                  .build_as_google_style()
                  .c_str());
}
}  // namespace internal
#endif
}  // namespace scaluq
