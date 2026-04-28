#include <gtest/gtest.h>

#include <scaluq/state/density_matrix.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class DensityMatrixTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(DensityMatrixTest, TestTypes, NameGenerator);

TYPED_TEST(DensityMatrixTest, HaarRandomStateTracePurity) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const int n_tries = 10;
    for (int n = 1; n <= n_tries; n++) {
        const auto state = DensityMatrix<Prec, Space>::Haar_random_state(n);
        ASSERT_NEAR(state.get_trace().real(), 1., eps<Prec>);
        ASSERT_NEAR(state.get_trace().imag(), 0., eps<Prec>);
        ASSERT_NEAR(state.get_purity(),
                    1.,
                    eps<Prec> * 10.);  // Large epsilon because the purity calculation involves
                                       // large number of floating-point operations.
    }
}

TYPED_TEST(DensityMatrixTest, CopyDensityMatrix) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const int n = 5;
    const auto state = DensityMatrix<Prec, Space>::Haar_random_state(n);
    DensityMatrix state_cp = state.copy();
    auto mat1 = state.get_matrix();
    auto mat2 = state_cp.get_matrix();
    ASSERT_EQ(mat1, mat2);
}

TYPED_TEST(DensityMatrixTest, ZeroNormState) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 5;

    DensityMatrix state(DensityMatrix<Prec, Space>::Haar_random_state(n));
    state.set_zero_norm_state();
    auto state_cp = state.get_matrix();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        for (std::uint64_t j = 0; j < state.dim(); ++j) {
            ASSERT_EQ((StdComplex)state_cp(i, j), StdComplex(0, 0));
        }
    }
}

TYPED_TEST(DensityMatrixTest, ComputationalBasisState) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 5;

    DensityMatrix state(DensityMatrix<Prec, Space>::Haar_random_state(n));
    state.set_computational_basis(31);
    auto state_cp = state.get_matrix();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        for (std::uint64_t j = 0; j < state.dim(); ++j) {
            if (i == 31 && j == 31) {
                ASSERT_EQ((StdComplex)state_cp(i, j), StdComplex(1, 0));
            } else {
                ASSERT_EQ((StdComplex)state_cp(i, j), StdComplex(0, 0));
            }
        }
    }
}

TYPED_TEST(DensityMatrixTest, HaarRandomStateSameSeed) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const int n_tries = 10;
    for (int n = 1; n <= n_tries; n++) {
        const auto state1 = DensityMatrix<Prec, Space>::Haar_random_state(n, 0);
        const auto state2 = DensityMatrix<Prec, Space>::Haar_random_state(n, 0);
        auto mat1 = state1.get_matrix();
        auto mat2 = state2.get_matrix();
        ASSERT_EQ(mat1, mat2);
    }
}

TYPED_TEST(DensityMatrixTest, HaarRandomStateWithoutSeed) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const int n_tries = 10;
    for (int n = 1; n <= n_tries; n++) {
        const auto state1 = DensityMatrix<Prec, Space>::Haar_random_state(n);
        const auto state2 = DensityMatrix<Prec, Space>::Haar_random_state(n);
        auto mat1 = state1.get_matrix();
        auto mat2 = state2.get_matrix();
        ASSERT_NE(mat1, mat2);
    }
}

TYPED_TEST(DensityMatrixTest, SpaceConversionCreatesIndependentCopy) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 5;
    auto state = DensityMatrix<Prec, Space>::Haar_random_state(n, 0);
    const auto original = state.get_matrix();

    auto state_default = state.copy_to_default_space();
    auto state_host = state.copy_to_host_space();

    auto default_mat = state_default.get_matrix();
    auto host_mat = state_host.get_matrix();
    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        for (std::uint64_t j = 0; j < state.dim(); ++j) {
            ASSERT_NEAR(default_mat(i, j).real(), original(i, j).real(), eps<Prec>);
            ASSERT_NEAR(default_mat(i, j).imag(), original(i, j).imag(), eps<Prec>);
            ASSERT_NEAR(host_mat(i, j).real(), original(i, j).real(), eps<Prec>);
            ASSERT_NEAR(host_mat(i, j).imag(), original(i, j).imag(), eps<Prec>);
        }
    }

    state_default.set_coherence_at(0, 0, StdComplex(0.25, -0.5));
    auto unchanged = state.get_coherence_at(0, 0);
    ASSERT_NEAR(unchanged.real(), original(0, 0).real(), eps<Prec>);
    ASSERT_NEAR(unchanged.imag(), original(0, 0).imag(), eps<Prec>);

    state_host.set_coherence_at(0, 0, StdComplex(0.25, -0.5));
    unchanged = state.get_coherence_at(0, 0);
    ASSERT_NEAR(unchanged.real(), original(0, 0).real(), eps<Prec>);
    ASSERT_NEAR(unchanged.imag(), original(0, 0).imag(), eps<Prec>);
}

TYPED_TEST(DensityMatrixTest, GetZeroProbability) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 5;
    auto state_vector = StateVector<Prec, Space>::Haar_random_state(n);
    DensityMatrix density_matrix(state_vector);
    for (std::uint64_t t = 0; t < n; t++) {
        ASSERT_NEAR(density_matrix.get_zero_probability(t),
                    state_vector.get_zero_probability(t),
                    eps<Prec>);
    }
}

TYPED_TEST(DensityMatrixTest, GetMarginalProbability) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 2;
    auto state_vector = StateVector<Prec, Space>::Haar_random_state(n);
    DensityMatrix density_matrix(state_vector);
    std::vector<std::uint64_t> measured_values(n, StateVector<Prec, Space>::UNMEASURED);
    for (std::uint64_t val0 :
         {std::uint64_t{0}, std::uint64_t{1}, StateVector<Prec, Space>::UNMEASURED}) {
        for (std::uint64_t val1 :
             {std::uint64_t{0}, std::uint64_t{1}, StateVector<Prec, Space>::UNMEASURED}) {
            measured_values[0] = val0;
            measured_values[1] = val1;
            ASSERT_NEAR(density_matrix.get_marginal_probability(measured_values),
                        state_vector.get_marginal_probability(measured_values),
                        eps<Prec>);
        }
    }
}

TYPED_TEST(DensityMatrixTest, GetComputationalBasisEntropy) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 5;
    auto state_vector = StateVector<Prec, Space>::Haar_random_state(n);
    DensityMatrix density_matrix(state_vector);
    ASSERT_NEAR(density_matrix.get_computational_basis_entropy(),
                state_vector.get_computational_basis_entropy(),
                eps<Prec>);
}

TYPED_TEST(DensityMatrixTest, SamplingSuperpositionState) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 10;
    const std::uint64_t nshot = 65536;
    const std::uint64_t test_count = 10;
    std::uint64_t pass_count = 0;
    for (std::uint64_t test_i = 0; test_i < test_count; test_i++) {
        StateVector<Prec, Space> state(n);
        state.set_computational_basis(0);
        for (std::uint64_t i = 1; i <= 4; ++i) {
            StateVector<Prec, Space> tmp_state(n);
            tmp_state.set_computational_basis(i);
            state.add_state_vector_with_coef(1 << i, tmp_state);
        }
        state.normalize();
        DensityMatrix<Prec, Space> dm(state);
        std::vector<std::uint64_t> res = dm.sampling(nshot);

        std::array<std::uint64_t, 5> cnt = {};
        for (std::uint64_t i = 0; i < nshot; ++i) {
            ASSERT_GE(res[i], 0);
            ASSERT_LE(res[i], (1 << n) - 1);
            cnt[res[i]] += 1;
        }
        bool pass = true;
        for (std::uint64_t i = 0; i < 4; i++) {
            std::string err_message = _CHECK_GT(cnt[i + 1], cnt[i]);
            if (err_message != "") {
                pass = false;
                std::cerr << err_message;
            }
        }
        if (pass) pass_count++;
    }
    ASSERT_GE(pass_count, test_count - 1);
}

TYPED_TEST(DensityMatrixTest, SamplingComputationalBasis) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 10;
    const std::uint64_t nshot = 1024;
    DensityMatrix<Prec, Space> state(n);
    state.set_computational_basis(42);
    std::vector<std::uint64_t> res = state.sampling(nshot);

    for (std::uint64_t i = 0; i < nshot; ++i) {
        ASSERT_EQ(res[i], 42);
    }
}

TYPED_TEST(DensityMatrixTest, GetPartialTrace) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    auto state = DensityMatrix<Prec, Space>::Haar_random_state(n);
    auto mat_state = state.get_matrix();
    auto partial_traced = state.get_partial_trace({0, 2});
    auto mat_partial_traced = partial_traced.get_matrix();
    ASSERT_EQ(partial_traced.n_qubits(), 2);
    int remained_base[] = {0, 2, 8, 10};
    for (std::uint64_t i = 0; i < partial_traced.dim(); ++i) {
        for (std::uint64_t j = 0; j < partial_traced.dim(); ++j) {
            StdComplex sum = StdComplex(0, 0);
            for (auto k : {0, 1, 4, 5}) {
                sum += mat_state(remained_base[i] + k, remained_base[j] + k);
            }
            ASSERT_NEAR(mat_partial_traced(i, j).real(), sum.real(), eps<Prec>);
            ASSERT_NEAR(mat_partial_traced(i, j).imag(), sum.imag(), eps<Prec>);
        }
    }
}

TYPED_TEST(DensityMatrixTest, AddDensityMatrixWithCoef) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 5;
    const double coef = 2.5;
    DensityMatrix state1(DensityMatrix<Prec, Space>::Haar_random_state(n));
    DensityMatrix state2(DensityMatrix<Prec, Space>::Haar_random_state(n));
    auto mat1 = state1.get_matrix();
    auto mat2 = state2.get_matrix();

    state1.add_density_matrix_with_coef(coef, state2);
    auto new_mat = state1.get_matrix();

    for (std::uint64_t i = 0; i < state1.dim(); ++i) {
        for (std::uint64_t j = 0; j < state1.dim(); ++j) {
            StdComplex res = new_mat(i, j), val = mat1(i, j) + coef * mat2(i, j);
            ASSERT_NEAR(res.real(), val.real(), eps<Prec>);
            ASSERT_NEAR(res.imag(), val.imag(), eps<Prec>);
        }
    }
}

TYPED_TEST(DensityMatrixTest, MultiplyCoef) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 5;
    const double coef = 0.5;
    DensityMatrix state(DensityMatrix<Prec, Space>::Haar_random_state(n));
    auto mat = state.get_matrix();

    state.multiply_coef(coef);
    auto new_mat = state.get_matrix();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        for (std::uint64_t j = 0; j < state.dim(); ++j) {
            StdComplex res = new_mat(i, j), val = coef * mat(i, j);
            ASSERT_NEAR(res.real(), val.real(), eps<Prec>);
            ASSERT_NEAR(res.imag(), val.imag(), eps<Prec>);
        }
    }
}
