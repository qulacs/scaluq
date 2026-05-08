#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../circuit/circuit.hpp"

namespace scaluq::qasm2 {

template <Precision Prec>
struct Qasm2Circuit {
    Circuit<Prec> circuit;
    std::uint64_t n_qubits = 0;
    std::uint64_t n_clbits = 0;
    std::vector<std::string> warnings;
};

template <Precision Prec>
Qasm2Circuit<Prec> loads(std::string_view source);

template <Precision Prec>
std::string dumps(const Circuit<Prec>& circuit, std::optional<std::uint64_t> n_qubits = std::nullopt);

}  // namespace scaluq::qasm2

#ifdef SCALUQ_USE_NANOBIND
#include "../types.hpp"

namespace scaluq::internal {
template <Precision Prec>
void bind_qasm2_hpp(nb::module_& m) {
    using namespace nb::literals;

    auto mqasm2 =
        m.def_submodule("qasm2", "Utilities for importing and exporting OpenQASM 2.0 programs.");
    nb::class_<qasm2::Qasm2Circuit<Prec>>(
        mqasm2,
        "Qasm2Circuit",
        DocString()
            .desc("Result object returned by :func:`loads`.")
            .desc("This object stores the Scaluq circuit produced from an OpenQASM 2.0 "
                  "program together with program-level metadata that is not stored in "
                  ":class:`Circuit` itself.")
            .note("`n_qubits` and `n_clbits` come from the declared QASM registers. They may be "
                  "larger than the number of qubits or classical bits touched by the gates.")
            .build_as_google_style()
            .c_str())
        .def_ro("circuit",
                &qasm2::Qasm2Circuit<Prec>::circuit,
                "The imported :class:`Circuit`.")
        .def_ro("n_qubits",
                &qasm2::Qasm2Circuit<Prec>::n_qubits,
                "Total number of qubits declared by OpenQASM `qreg` statements.")
        .def_ro("n_clbits",
                &qasm2::Qasm2Circuit<Prec>::n_clbits,
                "Total number of classical bits declared by OpenQASM `creg` statements.")
        .def_ro("warnings",
                &qasm2::Qasm2Circuit<Prec>::warnings,
                "Non-fatal import warnings, such as ignored barriers.");

    mqasm2.def(
        "loads",
        &qasm2::loads<Prec>,
        "source"_a,
        DocString()
            .desc("Parse an OpenQASM 2.0 program from a string.")
            .desc("Supported qelib1 operations are converted into Scaluq gates. Register "
                  "declarations are flattened into zero-based qubit and classical-bit indices.")
            .arg("source", "str", "OpenQASM 2.0 source code.")
            .ret("Qasm2Circuit",
                 "Imported circuit plus QASM-level metadata such as declared qubit count, "
                 "declared classical-bit count, and warnings.")
            .note("Measurements are imported as :func:`gate.Measurement` gates. Unsupported "
                  "features such as custom gate definitions, conditionals, and reset statements "
                  "raise `RuntimeError`.")
            .ex(DocString::Code({">>> result = qasm2.loads('OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; h q[0];')",
                                 ">>> result.n_qubits",
                                 "1",
                                 ">>> result.circuit.n_gates()",
                                 "1"}))
            .build_as_google_style()
            .c_str());
    mqasm2.def("dumps",
               &qasm2::dumps<Prec>,
               "circuit"_a,
               "n_qubits"_a = std::nullopt,
               DocString()
                   .desc("Serialize a Scaluq circuit as an OpenQASM 2.0 program.")
                   .desc("The output uses `qreg q[...]` and, when measurement gates are present, "
                         "`creg c[...]`. Supported controlled gates are emitted using qelib1 "
                         "names such as `cx`, `ccx`, `crz`, and `cswap`.")
                   .arg("circuit", "Circuit", "Circuit to serialize.")
                   .arg("n_qubits",
                        "int, optional",
                        true,
                        "Number of qubits to declare in the output `qreg`. If omitted, the "
                        "smallest value required by the circuit operands is used.")
                   .ret("str", "OpenQASM 2.0 source code.")
                   .note("Only gates representable by the supported OpenQASM 2.0/qelib1 subset "
                         "can be exported. Unsupported gates, control value 0, and "
                         "reset-after-measurement gates raise `RuntimeError`.")
                   .note("Parameter keys used by parametric rotation gates must be valid "
                         "OpenQASM identifiers because they are emitted into angle expressions.")
                   .ex(DocString::Code({">>> circuit = Circuit()",
                                        ">>> circuit.add_gate(gate.H(0))",
                                        ">>> print(qasm2.dumps(circuit, 1))",
                                        "OPENQASM 2.0;",
                                        "include \"qelib1.inc\";",
                                        "qreg q[1];",
                                        "h q[0];"}))
                   .build_as_google_style()
                   .c_str());
}
}  // namespace scaluq::internal
#endif
