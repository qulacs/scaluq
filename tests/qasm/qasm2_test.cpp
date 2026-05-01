#include <gtest/gtest.h>

#include <numbers>

#include <scaluq/all.hpp>

#include "../test_environment.hpp"

using namespace scaluq;

template <typename T>
class Qasm2Test : public FixtureBase<T> {};
TYPED_TEST_SUITE(Qasm2Test, TestTypes, NameGenerator);

TYPED_TEST(Qasm2Test, LoadsStandardGates) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::string source = R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg c[2];
        h q[0]; cx q[0], q[1];
        rx(pi / 2) q[2];
        u3(-pi/2, 0, 1.2e-3) q[3];
        ccx q[0], q[1], q[2];
        swap q[2], q[3];
        // measurement is intentionally unsupported for now.
    )";

    auto loaded = qasm2::loads<Prec>(source);
    EXPECT_EQ(loaded.n_qubits, 4);
    EXPECT_EQ(loaded.n_clbits, 2);
    EXPECT_EQ(loaded.circuit.n_gates(), 6);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(0)).gate_type(), GateType::H);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(1)).gate_type(), GateType::X);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(2)).gate_type(), GateType::RX);
}

TYPED_TEST(Qasm2Test, LoadsSymbolicRotationsAsParamGates) {
    constexpr Precision Prec = TestFixture::Prec;
    auto loaded = qasm2::loads<Prec>(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        rx(2 * theta) q[0];
        crz(phi / 2) q[0], q[1];
    )");

    EXPECT_EQ(loaded.circuit.n_gates(), 2);
    EXPECT_EQ(loaded.circuit.get_param_key_at(0), std::optional<std::string>("theta"));
    EXPECT_EQ(loaded.circuit.get_param_key_at(1), std::optional<std::string>("phi"));
}

TYPED_TEST(Qasm2Test, DumpsAndLoadsCircuit) {
    constexpr Precision Prec = TestFixture::Prec;
    Circuit<Prec> circuit;
    circuit.add_gate(gate::H<Prec>(0));
    circuit.add_gate(gate::CX<Prec>(0, 1));
    circuit.add_gate(gate::RZ<Prec>(1, std::numbers::pi / 4));
    circuit.add_param_gate(gate::ParamRX<Prec>(2, 0.5), "theta");

    const std::string dumped = qasm2::dumps(circuit, 3);
    EXPECT_NE(dumped.find("qreg q[3];"), std::string::npos);
    EXPECT_NE(dumped.find("cx q[0], q[1];"), std::string::npos);
    EXPECT_NE(dumped.find("rx(0.5*theta) q[2];"), std::string::npos);

    auto loaded = qasm2::loads<Prec>(dumped);
    EXPECT_EQ(loaded.n_qubits, 3);
    EXPECT_EQ(loaded.circuit.n_gates(), 4);
    EXPECT_EQ(loaded.circuit.get_param_key_at(3), std::optional<std::string>("theta"));
}

TYPED_TEST(Qasm2Test, RejectsUnsupportedMeasurement) {
    constexpr Precision Prec = TestFixture::Prec;
    EXPECT_THROW(
        (qasm2::loads<Prec>(R"(
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
        )")),
        std::runtime_error);
}

TYPED_TEST(Qasm2Test, RejectsOutOfRangeQubit) {
    constexpr Precision Prec = TestFixture::Prec;
    EXPECT_THROW(
        (qasm2::loads<Prec>(R"(
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            x q[1];
        )")),
        std::runtime_error);
}
