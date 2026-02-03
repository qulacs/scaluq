#include <gtest/gtest.h>

#include <scaluq/operator/pauli_operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class PauliOperatorTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(PauliOperatorTest, TestTypes, NameGenerator);

TYPED_TEST(PauliOperatorTest, ContainsExtraWhitespace) {
    constexpr Precision Prec = TestFixture::Prec;
    PauliOperator<Prec> expected = PauliOperator<Prec>("X 0", 1.0);
    PauliOperator<Prec> pauli_whitespace = PauliOperator<Prec>("X 0 ", 1.0);
    EXPECT_EQ(1, pauli_whitespace.target_qubit_list().size());
    EXPECT_EQ(1, pauli_whitespace.pauli_id_list().size());
    EXPECT_EQ(expected.get_pauli_string(), pauli_whitespace.get_pauli_string());
}

TYPED_TEST(PauliOperatorTest, EmptyStringConstructsIdentity) {
    constexpr Precision Prec = TestFixture::Prec;
    const auto identity = PauliOperator<Prec>("", 1.0);
    ASSERT_EQ(0, identity.target_qubit_list().size());
    ASSERT_EQ(0, identity.pauli_id_list().size());
    ASSERT_EQ("", identity.get_pauli_string());
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringA) {
    constexpr Precision Prec = TestFixture::Prec;
    double coef = 2.0;
    std::string Pauli_string = "X 0 X Z 1 Y 2";
    EXPECT_THROW((PauliOperator<Prec>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringB) {
    constexpr Precision Prec = TestFixture::Prec;
    double coef = 2.0;
    std::string Pauli_string = "X {i}";
    EXPECT_THROW((PauliOperator<Prec>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringC) {
    constexpr Precision Prec = TestFixture::Prec;
    double coef = 2.0;
    std::string Pauli_string = "X 4x";
    EXPECT_THROW((PauliOperator<Prec>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringD) {
    constexpr Precision Prec = TestFixture::Prec;
    double coef = 2.0;
    std::string Pauli_string = "4 X";
    EXPECT_THROW((PauliOperator<Prec>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, BrokenPauliStringE) {
    constexpr Precision Prec = TestFixture::Prec;
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y ";
    EXPECT_THROW((PauliOperator<Prec>(Pauli_string, coef)), std::runtime_error);
}

TYPED_TEST(PauliOperatorTest, SpacedPauliString) {
    constexpr Precision Prec = TestFixture::Prec;
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y 1 ";
    PauliOperator<Prec> pauli = PauliOperator<Prec>(Pauli_string, coef);
    size_t PauliSize = pauli.target_qubit_list().size();
    ASSERT_EQ(PauliSize, 2);
}

template <Precision Prec>
struct PauliTestParam {
    std::string test_name;
    PauliOperator<Prec> op1;
    PauliOperator<Prec> op2;
    PauliOperator<Prec> expected;

    PauliTestParam(const std::string& test_name_,
                   const PauliOperator<Prec>& op1_,
                   const PauliOperator<Prec>& op2_,
                   const PauliOperator<Prec>& expected_)
        : test_name(test_name_), op1(op1_), op2(op2_), expected(expected_) {}
};

template <Precision Prec>
std::ostream& operator<<(std::ostream& stream, const PauliTestParam<Prec>& p) {
    return stream << p.test_name;
}

#define TEST_FOR_PRECISION(PREC)                                                                  \
    class PauliOperatorMultiplyTest_##PREC                                                        \
        : public ::testing::TestWithParam<PauliTestParam<Precision::PREC>> {};                    \
                                                                                                  \
    TEST_P(PauliOperatorMultiplyTest_##PREC, MultiplyTest) {                                      \
        constexpr Precision Prec = Precision::PREC;                                               \
        const auto p = GetParam();                                                                \
        PauliOperator<Prec> res = p.op1 * p.op2;                                                  \
        EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());                         \
        EXPECT_EQ(p.expected.coef(), res.coef());                                                 \
    }                                                                                             \
                                                                                                  \
    TEST_P(PauliOperatorMultiplyTest_##PREC, MultiplyAssignmentTest) {                            \
        constexpr Precision Prec = Precision::PREC;                                               \
        const auto p = GetParam();                                                                \
        PauliOperator<Prec> res = p.op1;                                                          \
        res = res * p.op2;                                                                        \
        EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());                         \
        EXPECT_EQ(p.expected.coef(), res.coef());                                                 \
    }                                                                                             \
                                                                                                  \
    INSTANTIATE_TEST_CASE_P(                                                                      \
        SinglePauli,                                                                              \
        PauliOperatorMultiplyTest_##PREC,                                                         \
        testing::Values(PauliTestParam("XX",                                                      \
                                       PauliOperator<Precision::PREC>("X 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("X 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("I 0", 4.0)),               \
                        PauliTestParam("XY",                                                      \
                                       PauliOperator<Precision::PREC>("X 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Y 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Z 0", StdComplex(0, 4))),  \
                        PauliTestParam("XZ",                                                      \
                                       PauliOperator<Precision::PREC>("X 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Z 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Y 0", StdComplex(0, -4))), \
                        PauliTestParam("YX",                                                      \
                                       PauliOperator<Precision::PREC>("Y 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("X 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Z 0", StdComplex(0, -4))), \
                        PauliTestParam("YY",                                                      \
                                       PauliOperator<Precision::PREC>("Y 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Y 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("I 0", 4.0)),               \
                        PauliTestParam("YZ",                                                      \
                                       PauliOperator<Precision::PREC>("Y 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Z 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("X 0", StdComplex(0, 4))),  \
                        PauliTestParam("ZX",                                                      \
                                       PauliOperator<Precision::PREC>("Z 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("X 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Y 0", StdComplex(0, 4))),  \
                        PauliTestParam("ZY",                                                      \
                                       PauliOperator<Precision::PREC>("Z 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Y 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("X 0", StdComplex(0, -4))), \
                        PauliTestParam("ZZ",                                                      \
                                       PauliOperator<Precision::PREC>("Z 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Z 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("I 0", 4.0))),              \
        testing::PrintToStringParamName());                                                       \
                                                                                                  \
    INSTANTIATE_TEST_CASE_P(                                                                      \
        MultiPauli,                                                                               \
        PauliOperatorMultiplyTest_##PREC,                                                         \
        testing::Values(PauliTestParam("X_Y",                                                     \
                                       PauliOperator<Precision::PREC>("X 0", 2.0),                \
                                       PauliOperator<Precision::PREC>("Y 1", 2.0),                \
                                       PauliOperator<Precision::PREC>("X 0 Y 1", 4.0)),           \
                        PauliTestParam("XY_YX",                                                   \
                                       PauliOperator<Precision::PREC>("X 0 Y 1", 2.0),            \
                                       PauliOperator<Precision::PREC>("Y 0 X 1", 2.0),            \
                                       PauliOperator<Precision::PREC>("Z 0 Z 1", 4.0))),          \
        testing::PrintToStringParamName());                                                       \
                                                                                                  \
    INSTANTIATE_TEST_CASE_P(                                                                      \
        MultiPauliPauli,                                                                          \
        PauliOperatorMultiplyTest_##PREC,                                                         \
        ([]() {                                                                                   \
            double coef = 2.0;                                                                    \
            std::uint64_t MAX_TERM = 64;                                                          \
            std::string pauli_string_x = "";                                                      \
            std::string pauli_string_y = "";                                                      \
            std::string pauli_string_z = "";                                                      \
                                                                                                  \
            for (std::uint64_t i = 0; i < MAX_TERM; i++) {                                        \
                pauli_string_x += "X " + std::to_string(i);                                       \
                pauli_string_y += "Y " + std::to_string(i);                                       \
                pauli_string_z += "Z " + std::to_string(i);                                       \
                if (i + 1 < MAX_TERM) {                                                           \
                    pauli_string_x += " ";                                                        \
                    pauli_string_y += " ";                                                        \
                    pauli_string_z += " ";                                                        \
                }                                                                                 \
            }                                                                                     \
                                                                                                  \
            PauliOperator<Precision::PREC> expected =                                             \
                PauliOperator<Precision::PREC>(pauli_string_x, coef * coef);                      \
            PauliOperator<Precision::PREC> pauli_y =                                              \
                PauliOperator<Precision::PREC>(pauli_string_y, coef);                             \
            PauliOperator<Precision::PREC> pauli_z =                                              \
                PauliOperator<Precision::PREC>(pauli_string_z, coef);                             \
                                                                                                  \
            return testing::Values(PauliTestParam("Z_Y", pauli_z, pauli_y, expected));            \
        }()),                                                                                     \
        testing::PrintToStringParamName());
#ifdef SCALUQ_FLOAT16
TEST_FOR_PRECISION(F16)
#endif
#ifdef SCALUQ_FLOAT32
TEST_FOR_PRECISION(F32)
#endif
#ifdef SCALUQ_FLOAT64
TEST_FOR_PRECISION(F64)
#endif
#ifdef SCALUQ_BFLOAT16
TEST_FOR_PRECISION(BF16)
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

    PauliOperator<Prec> op(0b001, 0b010, StdComplex(2));
    op.apply_to_state(state_vector);
    std::vector<StdComplex> expected = {2, 0, -6, -4, 10, 8, -14, -12};
    ASSERT_EQ(state_vector.get_amplitudes(), expected);
}

TYPED_TEST(PauliOperatorTest, GetExpectationValueTest) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n_qubits = 5;
    const std::uint64_t n_batches = 10;
    const double coef = 2.0;
    const std::uint64_t n_pauli_gate_types = 4;
    std::default_random_engine engine(0);
    std::uniform_int_distribution<std::size_t> dist(0, 3);

    std::vector<std::uint64_t> indices(n_qubits);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<std::uint64_t> pauli_id_vec;
    std::vector<std::uint64_t> pauli_on(n_qubits);
    for (std::uint64_t target = 0; target < n_qubits; target++) {
        pauli_id_vec.emplace_back(dist(engine) % n_pauli_gate_types);
    }

    ComplexMatrix matrix;
    if (pauli_id_vec[0] == 0) {
        matrix = make_I();
    } else if (pauli_id_vec[0] == 1) {
        matrix = make_X();
    } else if (pauli_id_vec[0] == 2) {
        matrix = make_Y();
    } else if (pauli_id_vec[0] == 3) {
        matrix = make_Z();
    }
    for (int i = 1; i < (int)n_qubits; i++) {
        if (pauli_id_vec[i] == 0) {
            matrix = internal::kronecker_product(make_I(), matrix);
        } else if (pauli_id_vec[i] == 1) {
            matrix = internal::kronecker_product(make_X(), matrix);
        } else if (pauli_id_vec[i] == 2) {
            matrix = internal::kronecker_product(make_Y(), matrix);
        } else if (pauli_id_vec[i] == 3) {
            matrix = internal::kronecker_product(make_Z(), matrix);
        }
    }

    auto convert_to_eigen_vector = [&](StateVector<Prec, Space>& state) {
        auto amplitudes = state.get_amplitudes();
        ComplexVector state_eigen = ComplexVector::Zero(state.dim());
        for (auto i = std::size_t{0}; i < amplitudes.size(); i++) {
            state_eigen[i] = amplitudes[i];
        }
        return state_eigen;
    };

    StateVector<Prec, Space> state(n_qubits);
    StateVectorBatched<Prec, Space> states(n_batches, n_qubits);
    state.set_Haar_random_state(0);
    states.set_Haar_random_state(false, 0);

    // check get_expectation_value(StateVector)
    PauliOperator<Prec> pauli(indices, pauli_id_vec, coef);
    StdComplex res = pauli.get_expectation_value(state);
    ComplexVector eigen_state = convert_to_eigen_vector(state);
    StdComplex res_eigen = eigen_state.adjoint() * matrix * eigen_state;
    res_eigen *= coef;
    check_near<Prec>(res, res_eigen);

    // check get_expectation_value(BatchedStateVector)
    auto results = pauli.get_expectation_value(states);
    for (auto i = 0ULL; i < n_batches; i++) {
        auto sv = states.get_state_vector_at(i);
        eigen_state = convert_to_eigen_vector(sv);
        res_eigen = eigen_state.adjoint() * matrix * eigen_state;
        res_eigen *= coef;
        check_near<Prec>(results[i], res_eigen);
    }
}
