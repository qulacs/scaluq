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

    auto mqasm2 = m.def_submodule("qasm2", "OpenQASM 2.0 import/export utilities.");
    nb::class_<qasm2::Qasm2Circuit<Prec>>(mqasm2, "Qasm2Circuit")
        .def_ro("circuit", &qasm2::Qasm2Circuit<Prec>::circuit)
        .def_ro("n_qubits", &qasm2::Qasm2Circuit<Prec>::n_qubits)
        .def_ro("n_clbits", &qasm2::Qasm2Circuit<Prec>::n_clbits)
        .def_ro("warnings", &qasm2::Qasm2Circuit<Prec>::warnings);

    mqasm2.def("loads", &qasm2::loads<Prec>, "source"_a, "Load an OpenQASM 2.0 string.");
    mqasm2.def("dumps",
               &qasm2::dumps<Prec>,
               "circuit"_a,
               "n_qubits"_a = std::nullopt,
               "Dump a circuit as OpenQASM 2.0.");
}
}  // namespace scaluq::internal
#endif
