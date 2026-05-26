#include <gtest/gtest.h>

#include <numbers>

#include <scaluq/all.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

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
        measure q[3] -> c[1];
    )";

    auto loaded = qasm2::loads<Prec>(source);
    EXPECT_EQ(loaded.n_qubits, 4);
    EXPECT_EQ(loaded.n_clbits, 2);
    EXPECT_EQ(loaded.circuit.n_gates(), 7);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(0)).gate_type(), GateType::H);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(1)).gate_type(), GateType::X);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(2)).gate_type(), GateType::RX);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(6)).gate_type(), GateType::Measurement);
}

TYPED_TEST(Qasm2Test, RejectsSymbolicRotationAngles) {
    constexpr Precision Prec = TestFixture::Prec;
    EXPECT_THROW(
        (qasm2::loads<Prec>(R"(
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            rx(theta) q[0];
        )")),
        std::runtime_error);
}

TYPED_TEST(Qasm2Test, DumpsAndLoadsCircuit) {
    constexpr Precision Prec = TestFixture::Prec;
    Circuit<Prec> circuit;
    circuit.add_gate(gate::H<Prec>(0));
    circuit.add_gate(gate::CX<Prec>(0, 1));
    circuit.add_gate(gate::RZ<Prec>(1, std::numbers::pi / 4));
    circuit.add_gate(gate::Measurement<Prec>(1, 1));

    const std::string dumped = qasm2::dumps(circuit, 2);
    EXPECT_NE(dumped.find("qreg q[2];"), std::string::npos);
    EXPECT_NE(dumped.find("creg c[2];"), std::string::npos);
    EXPECT_NE(dumped.find("cx q[0], q[1];"), std::string::npos);
    EXPECT_NE(dumped.find("measure q[1] -> c[1];"), std::string::npos);

    auto loaded = qasm2::loads<Prec>(dumped);
    EXPECT_EQ(loaded.n_qubits, 2);
    EXPECT_EQ(loaded.n_clbits, 2);
    EXPECT_EQ(loaded.circuit.n_gates(), 4);
    EXPECT_EQ(std::get<0>(loaded.circuit.get_gate_at(3)).gate_type(), GateType::Measurement);
}

TYPED_TEST(Qasm2Test, DumpsAndLoadsControlledGatesWithControlValues) {
    constexpr Precision Prec = TestFixture::Prec;
    Circuit<Prec> circuit;
    circuit.add_gate(gate::X<Prec>(1, {0}, {1}));
    circuit.add_gate(gate::X<Prec>(3, {0, 2}, {1, 1}));
    circuit.add_gate(gate::Swap<Prec>(0, 1, {2}, {1}));

    const std::string dumped = qasm2::dumps(circuit, 4);
    EXPECT_NE(dumped.find("cx q[0], q[1];"), std::string::npos);
    EXPECT_NE(dumped.find("ccx q[0], q[2], q[3];"), std::string::npos);
    EXPECT_NE(dumped.find("cswap q[2], q[0], q[1];"), std::string::npos);

    auto loaded = qasm2::loads<Prec>(dumped);
    ASSERT_EQ(loaded.circuit.n_gates(), 3);
    const auto& cx = std::get<0>(loaded.circuit.get_gate_at(0));
    const auto& ccx = std::get<0>(loaded.circuit.get_gate_at(1));
    const auto& cswap = std::get<0>(loaded.circuit.get_gate_at(2));
    EXPECT_EQ(cx->control_qubit_list(), std::vector<std::uint64_t>{0});
    EXPECT_EQ(cx->control_value_list(), std::vector<std::uint64_t>{1});
    EXPECT_EQ(ccx->control_qubit_list(), (std::vector<std::uint64_t>{0, 2}));
    EXPECT_EQ(ccx->control_value_list(), (std::vector<std::uint64_t>{1, 1}));
    EXPECT_EQ(cswap->control_qubit_list(), std::vector<std::uint64_t>{2});
    EXPECT_EQ(cswap->control_value_list(), std::vector<std::uint64_t>{1});
}

TYPED_TEST(Qasm2Test, RejectsZeroControlValueExport) {
    constexpr Precision Prec = TestFixture::Prec;
    Circuit<Prec> circuit;
    circuit.add_gate(gate::X<Prec>(1, {0}, {0}));

    EXPECT_THROW(qasm2::dumps(circuit, 2), std::runtime_error);
}

TYPED_TEST(Qasm2Test, LoadedCircuitExecutionMatchesQasmSemantics) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    auto loaded = qasm2::loads<Prec>(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    )");

    StateVector<Prec, Space> state(loaded.n_qubits);
    ClassicalRegister classical_register(loaded.n_clbits);
    loaded.circuit.update_quantum_state(state, classical_register, {}, 0);

    EXPECT_TRUE(classical_register[0]);
    EXPECT_TRUE(classical_register[1]);
    const auto amplitudes = state.get_amplitudes();
    check_near<Prec>(amplitudes[0], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[2], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[3], StdComplex{1.0, 0.0});
}

TYPED_TEST(Qasm2Test, LoadsWholeRegisterMeasurement) {
    constexpr Precision Prec = TestFixture::Prec;
    auto loaded = qasm2::loads<Prec>(R"(
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        measure q -> c;
    )");

    EXPECT_EQ(loaded.n_qubits, 3);
    EXPECT_EQ(loaded.n_clbits, 3);
    ASSERT_EQ(loaded.circuit.n_gates(), 3);
    for (std::uint64_t i = 0; i < 3; ++i) {
        const auto& gate = std::get<0>(loaded.circuit.get_gate_at(i));
        EXPECT_EQ(gate.gate_type(), GateType::Measurement);
        EXPECT_EQ(MeasurementGate<Prec>(gate)->classical_bit_index(), i);
        EXPECT_EQ(gate->target_qubit_list(), std::vector<std::uint64_t>{i});
    }
}

TYPED_TEST(Qasm2Test, DumpsAreStableAfterReload) {
    constexpr Precision Prec = TestFixture::Prec;
    Circuit<Prec> circuit;
    circuit.add_gate(gate::H<Prec>(0));
    circuit.add_gate(gate::CX<Prec>(0, 1));
    circuit.add_gate(gate::RZ<Prec>(1, std::numbers::pi / 4));
    circuit.add_gate(gate::Measurement<Prec>(1, 1));

    const std::string first_dump = qasm2::dumps(circuit, 2);
    auto loaded = qasm2::loads<Prec>(first_dump);
    const std::string second_dump = qasm2::dumps(loaded.circuit, loaded.n_qubits);

    EXPECT_EQ(second_dump, first_dump);
}

TYPED_TEST(Qasm2Test, RejectsParametricGateExport) {
    constexpr Precision Prec = TestFixture::Prec;
    Circuit<Prec> circuit;
    circuit.add_param_gate(gate::ParamRX<Prec>(0, 0.5), "theta");

    EXPECT_THROW(qasm2::dumps(circuit, 1), std::runtime_error);
}

TYPED_TEST(Qasm2Test, RejectsZeroQubitExport) {
    constexpr Precision Prec = TestFixture::Prec;
    Circuit<Prec> empty_circuit;
    Circuit<Prec> identity_only_circuit;
    identity_only_circuit.add_gate(gate::I<Prec>());

    EXPECT_THROW(qasm2::dumps(empty_circuit), std::runtime_error);
    EXPECT_THROW(qasm2::dumps(empty_circuit, 0), std::runtime_error);
    EXPECT_THROW(qasm2::dumps(identity_only_circuit), std::runtime_error);
    EXPECT_NE(qasm2::dumps(identity_only_circuit, 1).find("qreg q[1];"), std::string::npos);
}

TYPED_TEST(Qasm2Test, RejectsMissingClassicalRegisterMeasurement) {
    constexpr Precision Prec = TestFixture::Prec;
    EXPECT_THROW(
        (qasm2::loads<Prec>(R"(
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            measure q[0] -> c[0];
        )")),
        std::runtime_error);
}

TYPED_TEST(Qasm2Test, RejectsMismatchedRegisterMeasurement) {
    constexpr Precision Prec = TestFixture::Prec;
    EXPECT_THROW(
        (qasm2::loads<Prec>(R"(
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[3];
            measure q -> c;
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
