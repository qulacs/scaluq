#include <gtest/gtest.h>

#include <scaluq/operator/pauli_operator.hpp>

#include "../test_environment.hpp"

using namespace scaluq;

#define FLOAT_AND_SPACE(Fp, Sp) template <std::floating_point Fp, ExecutionSpace Sp>
#define EXECUTE_TEST(Name)                  \
    TEST(PauliOperatorTest, Name) {         \
        Test##Name<double, DefaultSpace>(); \
        Test##Name<double, HostSpace>();    \
        Test##Name<float, DefaultSpace>();  \
        Test##Name<float, HostSpace>();     \
    }

FLOAT_AND_SPACE(Fp, Sp)
void TestContainsExtraWhitespace() {
    PauliOperator<Fp, Sp> expected("X 0", 1.0), pauli_whitespace("X 0 ", 1.0);

    EXPECT_EQ(1, pauli_whitespace.target_qubit_list().size());
    EXPECT_EQ(1, pauli_whitespace.pauli_id_list().size());
    EXPECT_EQ(expected.get_pauli_string(), pauli_whitespace.get_pauli_string());
}
EXECUTE_TEST(ContainsExtraWhitespace)

FLOAT_AND_SPACE(Fp, Sp)
void TestEmptyStringConstructsIdentity() {
    const auto identity = PauliOperator<Fp, Sp>("", 1.0);
    ASSERT_EQ(0, identity.target_qubit_list().size());
    ASSERT_EQ(0, identity.pauli_id_list().size());
    ASSERT_EQ("", identity.get_pauli_string());
}
EXECUTE_TEST(EmptyStringConstructsIdentity)

FLOAT_AND_SPACE(Fp, Sp)
void TestPauliQubitOverflow() {
    int n = 2;
    double coef = 2.0;
    std::string pauli_string = "X 0 X 1 X 3";
    PauliOperator<Fp, Sp> pauli(pauli_string, coef);
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    EXPECT_THROW((void)pauli.get_expectation_value(state), std::runtime_error);
}
EXECUTE_TEST(PauliQubitOverflow)

FLOAT_AND_SPACE(Fp, Sp)
void TestBrokenPauliString() {
    Fp coef = 2.0;
    std::string pauli_string1 = "4 X";
    std::string pauli_string2 = "X {i}";
    std::string pauli_string3 = "X 0 Y ";
    using PO = PauliOperator<Fp, Sp>;
    ASSERT_THROW(PO(pauli_string1, coef), std::runtime_error);
    ASSERT_THROW(PO(pauli_string2, coef), std::runtime_error);
    ASSERT_THROW(PO(pauli_string3, coef), std::runtime_error);
}
EXECUTE_TEST(BrokenPauliString)

FLOAT_AND_SPACE(Fp, Sp)
void TestSpacedPauliString() {
    Fp coef = 2.0;
    std::string pauli_string = "X 0 Y 1 ";
    PauliOperator<Fp, Sp> pauli(pauli_string, coef);
    size_t PauliSize = pauli.target_qubit_list().size();
    ASSERT_EQ(PauliSize, 2);
}
EXECUTE_TEST(SpacedPauliString)

struct PauliTestParam {
    std::string test_name;
    PauliOperator<double, DefaultSpace> op1, op2, expected;
    PauliOperator<double, HostSpace> op1_cpu, op2_cpu, expected_cpu;
    PauliOperator<float, DefaultSpace> op1_f, op2_f, expected_f;
    PauliOperator<float, HostSpace> op1_f_cpu, op2_f_cpu, expected_f_cpu;
    using Clx = StdComplex<double>;
    using Clx_f = StdComplex<float>;

    PauliTestParam(const std::string& _test_name,
                   const std::tuple<std::string, double, double>& _op1,  // pauli_string, real, imag
                   const std::tuple<std::string, double, double>& _op2,
                   const std::tuple<std::string, double, double>& _exp)
        : test_name(_test_name),
          op1(std::get<0>(_op1), Clx(std::get<1>(_op1), std::get<2>(_op1))),
          op2(std::get<0>(_op2), Clx(std::get<1>(_op2), std::get<2>(_op2))),
          expected(std::get<0>(_exp), Clx(std::get<1>(_exp), std::get<2>(_exp))),
          op1_cpu(std::get<0>(_op1), Clx(std::get<1>(_op1), std::get<2>(_op1))),
          op2_cpu(std::get<0>(_op2), Clx(std::get<1>(_op2), std::get<2>(_op2))),
          expected_cpu(std::get<0>(_exp), Clx(std::get<1>(_exp), std::get<2>(_exp))),
          op1_f(std::get<0>(_op1), Clx_f(std::get<1>(_op1), std::get<2>(_op1))),
          op2_f(std::get<0>(_op2), Clx_f(std::get<1>(_op2), std::get<2>(_op2))),
          expected_f(std::get<0>(_exp), Clx_f(std::get<1>(_exp), std::get<2>(_exp))),
          op1_f_cpu(std::get<0>(_op1), Clx_f(std::get<1>(_op1), std::get<2>(_op1))),
          op2_f_cpu(std::get<0>(_op2), Clx_f(std::get<1>(_op2), std::get<2>(_op2))),
          expected_f_cpu(std::get<0>(_exp), Clx_f(std::get<1>(_exp), std::get<2>(_exp))) {}

    friend std::ostream& operator<<(std::ostream& stream, const PauliTestParam& p) {
        return stream << p.test_name;
    }
};

class PauliOperatorMultiplyTest : public testing::TestWithParam<PauliTestParam> {};

TEST_P(PauliOperatorMultiplyTest, Multiply) {
    const auto p = GetParam();
    PauliOperator<double, DefaultSpace> res = p.op1 * p.op2;
    EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());
    EXPECT_EQ(p.expected.coef(), res.coef());
}

INSTANTIATE_TEST_CASE_P(
    SinglePauli,
    PauliOperatorMultiplyTest,
    testing::Values(PauliTestParam("XX", {"X 0", 2.0, 0.0}, {"X 0", 2.0, 0.0}, {"I 0", 4.0, 0.0}),
                    PauliTestParam("XY", {"X 0", 2.0, 0.0}, {"Y 0", 2.0, 0.0}, {"Z 0", 0.0, 4.0}),
                    PauliTestParam("XZ", {"X 0", 2.0, 0.0}, {"Z 0", 2.0, 0.0}, {"Y 0", 0.0, -4.0}),
                    PauliTestParam("YX", {"Y 0", 2.0, 0.0}, {"X 0", 2.0, 0.0}, {"Z 0", 0.0, -4.0}),
                    PauliTestParam("YY", {"Y 0", 2.0, 0.0}, {"Y 0", 2.0, 0.0}, {"I 0", 4.0, 0.0}),
                    PauliTestParam("YZ", {"Y 0", 2.0, 0.0}, {"Z 0", 2.0, 0.0}, {"X 0", 0.0, 4.0}),
                    PauliTestParam("ZX", {"Z 0", 2.0, 0.0}, {"X 0", 2.0, 0.0}, {"Y 0", 0.0, 4.0}),
                    PauliTestParam("ZY", {"Z 0", 2.0, 0.0}, {"Y 0", 2.0, 0.0}, {"X 0", 0.0, -4.0}),
                    PauliTestParam("ZZ", {"Z 0", 2.0, 0.0}, {"Z 0", 2.0, 0.0}, {"I 0", 4.0, 0.0})),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_CASE_P(MultiPauli,
                        PauliOperatorMultiplyTest,
                        testing::Values(PauliTestParam("X_Y",
                                                       {"X 0", 2.0, 0.0},
                                                       {"Y 1", 2.0, 0.0},
                                                       {"X 0 Y 1", 4.0, 0.0}),
                                        PauliTestParam("XY_YX",
                                                       {"X 0 Y 1", 2.0, 0.0},
                                                       {"Y 0 X 1", 2.0, 0.0},
                                                       {"Z 0 Z 1", 4.0, 0.0})),
                        testing::PrintToStringParamName());

INSTANTIATE_TEST_CASE_P(
    LongPauli,
    PauliOperatorMultiplyTest,
    [] {
        double coef = 2.0;
        std::uint64_t MAX_TERM = 64;
        std::string pauli_string_x = "";
        std::string pauli_string_y = "";
        std::string pauli_string_z = "";

        for (std::uint64_t i = 0; i < MAX_TERM; i++) {
            pauli_string_x += "X " + std::to_string(i);
            pauli_string_y += "Y " + std::to_string(i);
            pauli_string_z += "Z " + std::to_string(i);
            if (i + 1 < MAX_TERM) {
                pauli_string_x += " ";
                pauli_string_y += " ";
                pauli_string_z += " ";
            }
        }

        return testing::Values(PauliTestParam("Z_Y",
                                              {pauli_string_z, coef, 0.0},
                                              {pauli_string_y, coef, 0.0},
                                              {pauli_string_x, coef * coef, 0.0}));
    }(),
    testing::PrintToStringParamName());

FLOAT_AND_SPACE(Fp, Sp)
void TestApplyToState() {
    const std::uint64_t n_qubits = 3;
    StateVector<Fp, Sp> state_vector(n_qubits);
    state_vector.load([n_qubits] {
        std::vector<Complex<Fp>> tmp(1 << n_qubits);
        for (std::uint64_t i = 0; i < tmp.size(); ++i) tmp[i] = Complex<Fp>(i, 0);
        return tmp;
    }());

    PauliOperator<Fp, Sp> op(0b001, 0b010, StdComplex<Fp>(2));
    op.apply_to_state(state_vector);
    std::vector<Complex<Fp>> expected = {2, 0, -6, -4, 10, 8, -14, -12};
    ASSERT_EQ(state_vector.get_amplitudes(), expected);
}
EXECUTE_TEST(ApplyToState)
