#include <gtest/gtest.h>

#include <array>
#include <scaluq/operator/pauli_operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class PauliFullMatrixTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(PauliFullMatrixTest, TestTypes, NameGenerator);

TYPED_TEST(PauliFullMatrixTest, EmbeddedCoefficientsAndExplicitUnscaledMatrix) {
    constexpr auto Prec = TestFixture::Prec;
    const std::vector<StdComplex> coefficients{0., 2., -2., {.5, -.25}};
    const std::array<ComplexMatrix, 3> single{make_X(), make_Y(), make_Z()};
    const std::array<std::string, 3> names{"X 1", "Y 1", "Z 1"};
    for (std::size_t axis = 0; axis < names.size(); ++axis) {
        ComplexMatrix expected = internal::kronecker_product(
            make_I(), internal::kronecker_product(single[axis], make_I()));
        for (const auto& coef : coefficients) {
            PauliOperator<Prec> pauli(names[axis], coef);
            const auto full = pauli.get_full_matrix(3);
            const auto unscaled = pauli.get_full_matrix_ignoring_coef(3);
            ASSERT_EQ(full.rows(), 8);
            ASSERT_EQ(full.cols(), 8);
            for (Eigen::Index i = 0; i < 8; ++i) {
                for (Eigen::Index j = 0; j < 8; ++j) {
                    EXPECT_EQ(full(i, j), coef * expected(i, j));
                    EXPECT_EQ(unscaled(i, j), expected(i, j));
                }
            }
        }
    }
}

TYPED_TEST(PauliFullMatrixTest, ZeroQubitIdentityAndCompactMatrix) {
    constexpr auto Prec = TestFixture::Prec;
    for (const StdComplex coef : {StdComplex(0.), StdComplex(-2.), StdComplex(.5, -.25)}) {
        PauliOperator<Prec> identity("", coef);
        const auto scalar = identity.get_full_matrix(0);
        ASSERT_EQ(scalar.rows(), 1);
        ASSERT_EQ(scalar.cols(), 1);
        EXPECT_EQ(scalar(0, 0), coef);
        EXPECT_EQ(identity.get_full_matrix_ignoring_coef(0)(0, 0), StdComplex(1.));
        PauliOperator<Prec> pauli("Y 0 Z 1", coef);
        const auto full = pauli.get_full_matrix(2);
        const auto compact = pauli.get_matrix();
        for (Eigen::Index i = 0; i < full.rows(); ++i) {
            for (Eigen::Index j = 0; j < full.cols(); ++j) {
                EXPECT_EQ(full(i, j), compact(i, j));
            }
        }
    }
}
