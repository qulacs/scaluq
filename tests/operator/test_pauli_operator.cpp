#include <gtest/gtest.h>

#include <scaluq/operator/pauli_operator.hpp>

#include "../test_environment.hpp"

using namespace scaluq;

template <typename T>
class PauliOperatorTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(PauliOperatorTest, TestTypes, NameGenerator);

TYPED_TEST(PauliOperatorTest, ContainsExtraWhitespace) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    PauliOperator<Prec, Space> expected = PauliOperator<Prec, Space>("X 0", 1.0);
    PauliOperator<Prec, Space> pauli_whitespace = PauliOperator<Prec, Space>("X 0 ", 1.0);

    EXPECT_EQ(1, pauli_whitespace.target_qubit_list().size());
    EXPECT_EQ(1, pauli_whitespace.pauli_id_list().size());
    EXPECT_EQ(expected.get_pauli_string(), pauli_whitespace.get_pauli_string());
}

TYPED_TEST(PauliOperatorTest, EmptyStringConstructsIdentity) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const auto identity = PauliOperator<Prec, Space>("", 1.0);
    ASSERT_EQ(0, identity.target_qubit_list().size());
    ASSERT_EQ(0, identity.pauli_id_list().size());
    ASSERT_EQ("", identity.get_pauli_string());
}

TYPED_TEST(PauliOperatorTest, PauliQubitOverflow) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    int n = 2;
    double coef = 2.0;
    std::string Pauli_string = "X 0 X 1 X 3";
    PauliOperator<Prec, Space> pauli = PauliOperator<Prec, Space>(Pauli_string, coef);
    StateVector state = StateVector<Prec, Space>::Haar_random_state(n);
    EXPECT_THROW((void)pauli.get_expectation_value(state), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringA) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    double coef = 2.0;
    std::string Pauli_string = "X 0 X Z 1 Y 2";
    EXPECT_THROW((PauliOperator<Prec, Space>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringB) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    double coef = 2.0;
    std::string Pauli_string = "X {i}";
    EXPECT_THROW((PauliOperator<Prec, Space>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringC) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    double coef = 2.0;
    std::string Pauli_string = "X 4x";
    EXPECT_THROW((PauliOperator<Prec, Space>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringD) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    double coef = 2.0;
    std::string Pauli_string = "4 X";
    EXPECT_THROW((PauliOperator<Prec, Space>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, SpacedPauliString) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y 1 ";
    PauliOperator<Prec, Space> pauli = PauliOperator<Prec, Space>(Pauli_string, coef);
    size_t PauliSize = pauli.target_qubit_list().size();
    ASSERT_EQ(PauliSize, 2);
}

TYPED_TEST(PauliOperatorTest, PartedPauliString) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y ";
    EXPECT_THROW((PauliOperator<Prec, Space>(Pauli_string, coef)), std::runtime_error);
}

template <Precision Prec, ExecutionSpace Space>
struct PauliTestParam {
    std::string test_name;
    PauliOperator<Prec, Space> op1;
    PauliOperator<Prec, Space> op2;
    PauliOperator<Prec, Space> expected;

    PauliTestParam(const std::string& test_name_,
                   const PauliOperator<Prec, Space>& op1_,
                   const PauliOperator<Prec, Space>& op2_,
                   const PauliOperator<Prec, Space>& expected_)
        : test_name(test_name_), op1(op1_), op2(op2_), expected(expected_) {}
};

template <Precision Prec, ExecutionSpace Space>
std::ostream& operator<<(std::ostream& stream, const PauliTestParam<Prec, Space>& p) {
    return stream << p.test_name;
}

#define TEST_FOR_PRECISION_AND_SPACE(PREC, SPACE)                                                  \
    class PauliOperatorMultiplyTest_##PREC##SPACE                                                  \
        : public ::testing::TestWithParam<                                                         \
              PauliTestParam<Precision::PREC, ExecutionSpace::SPACE>> {};                          \
                                                                                                   \
    TEST_P(PauliOperatorMultiplyTest_##PREC##SPACE, MultiplyTest) {                                \
        constexpr Precision Prec = Precision::PREC;                                                \
        constexpr ExecutionSpace Space = ExecutionSpace::SPACE;                                    \
        const auto p = GetParam();                                                                 \
        PauliOperator<Prec, Space> res = p.op1 * p.op2;                                            \
        EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());                          \
        EXPECT_EQ(p.expected.coef(), res.coef());                                                  \
    }                                                                                              \
                                                                                                   \
    TEST_P(PauliOperatorMultiplyTest_##PREC##SPACE, MultiplyAssignmentTest) {                      \
        constexpr Precision Prec = Precision::PREC;                                                \
        constexpr ExecutionSpace Space = ExecutionSpace::SPACE;                                    \
        const auto p = GetParam();                                                                 \
        PauliOperator<Prec, Space> res = p.op1;                                                    \
        res = res * p.op2;                                                                         \
        EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());                          \
        EXPECT_EQ(p.expected.coef(), res.coef());                                                  \
    }                                                                                              \
                                                                                                   \
    INSTANTIATE_TEST_CASE_P(                                                                       \
        SinglePauli,                                                                               \
        PauliOperatorMultiplyTest_##PREC##SPACE,                                                   \
        testing::Values(                                                                           \
            PauliTestParam("XX",                                                                   \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("I 0", 4.0)),     \
            PauliTestParam(                                                                        \
                "XY",                                                                              \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", StdComplex(0, 4))),   \
            PauliTestParam(                                                                        \
                "XZ",                                                                              \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", StdComplex(0, -4))),  \
            PauliTestParam(                                                                        \
                "YX",                                                                              \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", StdComplex(0, -4))),  \
            PauliTestParam("YY",                                                                   \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("I 0", 4.0)),     \
            PauliTestParam(                                                                        \
                "YZ",                                                                              \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", StdComplex(0, 4))),   \
            PauliTestParam(                                                                        \
                "ZX",                                                                              \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", StdComplex(0, 4))),   \
            PauliTestParam(                                                                        \
                "ZY",                                                                              \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0", 2.0),                 \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", StdComplex(0, -4))),  \
            PauliTestParam("ZZ",                                                                   \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("I 0", 4.0))),    \
        testing::PrintToStringParamName());                                                        \
                                                                                                   \
    INSTANTIATE_TEST_CASE_P(                                                                       \
        MultiPauli,                                                                                \
        PauliOperatorMultiplyTest_##PREC##SPACE,                                                   \
        testing::Values(                                                                           \
            PauliTestParam("X_Y",                                                                  \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 1", 2.0),      \
                           PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0 Y 1", 4.0)), \
            PauliTestParam(                                                                        \
                "XY_YX",                                                                           \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("X 0 Y 1", 2.0),             \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Y 0 X 1", 2.0),             \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>("Z 0 Z 1", 4.0))),           \
        testing::PrintToStringParamName());                                                        \
                                                                                                   \
    INSTANTIATE_TEST_CASE_P(                                                                       \
        MultiPauliPauli,                                                                           \
        PauliOperatorMultiplyTest_##PREC##SPACE,                                                   \
        ([]() {                                                                                    \
            double coef = 2.0;                                                                     \
            std::uint64_t MAX_TERM = 64;                                                           \
            std::string pauli_string_x = "";                                                       \
            std::string pauli_string_y = "";                                                       \
            std::string pauli_string_z = "";                                                       \
                                                                                                   \
            for (std::uint64_t i = 0; i < MAX_TERM; i++) {                                         \
                pauli_string_x += "X " + std::to_string(i);                                        \
                pauli_string_y += "Y " + std::to_string(i);                                        \
                pauli_string_z += "Z " + std::to_string(i);                                        \
                if (i + 1 < MAX_TERM) {                                                            \
                    pauli_string_x += " ";                                                         \
                    pauli_string_y += " ";                                                         \
                    pauli_string_z += " ";                                                         \
                }                                                                                  \
            }                                                                                      \
                                                                                                   \
            PauliOperator<Precision::PREC, ExecutionSpace::SPACE> expected =                       \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>(pauli_string_x,              \
                                                                      coef * coef);                \
            PauliOperator<Precision::PREC, ExecutionSpace::SPACE> pauli_y =                        \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>(pauli_string_y, coef);       \
            PauliOperator<Precision::PREC, ExecutionSpace::SPACE> pauli_z =                        \
                PauliOperator<Precision::PREC, ExecutionSpace::SPACE>(pauli_string_z, coef);       \
                                                                                                   \
            return testing::Values(PauliTestParam("Z_Y", pauli_z, pauli_y, expected));             \
        }()),                                                                                      \
        testing::PrintToStringParamName());
#ifdef SCALUQ_FLOAT16
TEST_FOR_PRECISION_AND_SPACE(F16, Host)
#endif
#ifdef SCALUQ_FLOAT32
TEST_FOR_PRECISION_AND_SPACE(F32, Host)
#endif
#ifdef SCALUQ_FLOAT64
TEST_FOR_PRECISION_AND_SPACE(F64, Host)
#endif
#ifdef SCALUQ_BFLOAT16
TEST_FOR_PRECISION_AND_SPACE(BF16, Host)
#endif
#ifdef SCALUQ_USE_CUDA
#ifdef SCALUQ_FLOAT16
TEST_FOR_PRECISION_AND_SPACE(F16, Default)
#endif
#ifdef SCALUQ_FLOAT32
TEST_FOR_PRECISION_AND_SPACE(F32, Default)
#endif
#ifdef SCALUQ_FLOAT64
TEST_FOR_PRECISION_AND_SPACE(F64, Default)
#endif
#ifdef SCALUQ_BFLOAT16
TEST_FOR_PRECISION_AND_SPACE(BF16, Default)
#endif
#endif

TYPED_TEST(PauliOperatorTest, ApplyToStateTest) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n_qubits = 3;
    StateVector<Prec, Space> state_vector(n_qubits);
    state_vector.load([n_qubits] {
        std::vector<StdComplex> tmp(1 << n_qubits);
        for (std::uint64_t i = 0; i < tmp.size(); ++i) tmp[i] = StdComplex(i, 0);
        return tmp;
    }());

    PauliOperator<Prec, Space> op(0b001, 0b010, StdComplex(2));
    op.apply_to_state(state_vector);
    std::vector<StdComplex> expected = {2, 0, -6, -4, 10, 8, -14, -12};
    ASSERT_EQ(state_vector.get_amplitudes(), expected);
}
