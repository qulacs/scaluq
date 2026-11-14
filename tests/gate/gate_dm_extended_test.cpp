// Density-matrix coverage for every gate kind and every control pattern, checked against an
// Eigen reference U ρ U†.

#include <gtest/gtest.h>

#include <Eigen/QR>
#include <random>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/state/density_matrix.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class DMExtendedTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(DMExtendedTest, TestTypes, NameGenerator);

namespace {
ComplexMatrix random_unitary(std::uint64_t k, std::mt19937_64& rng) {
    std::normal_distribution<double> nd;
    const std::uint64_t d = 1ULL << k;
    ComplexMatrix m(d, d);
    for (std::uint64_t i = 0; i < d; ++i)
        for (std::uint64_t j = 0; j < d; ++j) m(i, j) = StdComplex(nd(rng), nd(rng));
    Eigen::HouseholderQR<ComplexMatrix> qr(m);
    return qr.householderQ();
}

ComplexMatrix random_matrix(std::uint64_t k, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0., 0.3);
    const std::uint64_t d = 1ULL << k;
    ComplexMatrix m(d, d);
    for (std::uint64_t i = 0; i < d; ++i)
        for (std::uint64_t j = 0; j < d; ++j) m(i, j) = StdComplex(nd(rng), nd(rng));
    return m;
}

std::vector<std::uint64_t> all_qubits(std::uint64_t n) {
    std::vector<std::uint64_t> v(n);
    for (std::uint64_t i = 0; i < n; ++i) v[i] = i;
    return v;
}

template <Precision Prec>
ComplexMatrix full_matrix(const Gate<Prec>& gate, std::uint64_t n) {
    auto qubits = all_qubits(n);
    return internal::get_expanded_matrix(gate->get_matrix(),
                                         gate->target_qubit_list(),
                                         gate->control_qubit_mask(),
                                         gate->control_value_mask(),
                                         qubits);
}

template <Precision Prec>
void expect_matrix_near(const ComplexMatrix& actual,
                        const ComplexMatrix& expected,
                        double tolerance,
                        const std::string& what) {
    ASSERT_EQ(actual.rows(), expected.rows());
    for (Eigen::Index i = 0; i < actual.rows(); ++i)
        for (Eigen::Index j = 0; j < actual.cols(); ++j)
            ASSERT_NEAR(std::abs(actual(i, j) - expected(i, j)), 0., tolerance)
                << what << " at (" << i << ", " << j << ")";
}

// ρ → U ρ U† against an Eigen reference built from the gate's own matrix.
template <Precision Prec, ExecutionSpace Space>
void check_against_matrix(std::uint64_t n, const Gate<Prec>& gate, const std::string& what) {
    auto dm = DensityMatrix<Prec, Space>::Haar_random_state(n);
    const ComplexMatrix rho = dm.get_matrix();
    gate->update_quantum_state(dm);
    const ComplexMatrix u = full_matrix(gate, n);
    expect_matrix_near<Prec>(dm.get_matrix(), u * rho * u.adjoint(), eps<Prec> * 10, what);
}

// Control patterns exercised for every gate: none, one control (value 0 / 1), two controls.
std::vector<std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>>> control_patterns(
    std::uint64_t a, std::uint64_t b) {
    return {{{}, {}}, {{a}, {1}}, {{a}, {0}}, {{a, b}, {1, 0}}, {{b, a}, {1, 1}}};
}
}  // namespace

TYPED_TEST(DMExtendedTest, StandardGatesAllControlPatterns) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 6;
    for (std::uint64_t t : {0ULL, 3ULL, 5ULL}) {
        const std::uint64_t c1 = (t + 1) % n, c2 = (t + 4) % n;
        for (const auto& [ctrl, val] : control_patterns(c1, c2)) {
            const std::vector<std::pair<std::string, Gate<Prec>>> gates = {
                {"X", gate::X<Prec>(t, ctrl, val)},
                {"Y", gate::Y<Prec>(t, ctrl, val)},
                {"Z", gate::Z<Prec>(t, ctrl, val)},
                {"H", gate::H<Prec>(t, ctrl, val)},
                {"S", gate::S<Prec>(t, ctrl, val)},
                {"Tdag", gate::Tdag<Prec>(t, ctrl, val)},
                {"SqrtX", gate::SqrtX<Prec>(t, ctrl, val)},
                {"SqrtYdag", gate::SqrtYdag<Prec>(t, ctrl, val)},
                {"P0", gate::P0<Prec>(t, ctrl, val)},
                {"P1", gate::P1<Prec>(t, ctrl, val)},
                {"RX", gate::RX<Prec>(t, 0.7, ctrl, val)},
                {"RY", gate::RY<Prec>(t, -1.1, ctrl, val)},
                {"RZ", gate::RZ<Prec>(t, 0.4, ctrl, val)},
                {"U1", gate::U1<Prec>(t, 0.3, ctrl, val)},
                {"U3", gate::U3<Prec>(t, 0.3, -0.2, 1.4, ctrl, val)},
                {"GlobalPhase", gate::GlobalPhase<Prec>(0.8, ctrl, val)},
            };
            for (const auto& [name, g] : gates) {
                check_against_matrix<Prec, Space>(
                    n,
                    g,
                    name + " t=" + std::to_string(t) + " #ctrl=" + std::to_string(ctrl.size()));
            }
        }
    }
}

TYPED_TEST(DMExtendedTest, TwoQubitGatesAllControlPatterns) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 6;
    for (const auto& [a, b] :
         std::vector<std::pair<std::uint64_t, std::uint64_t>>{{0, 1}, {4, 1}, {5, 2}}) {
        const std::uint64_t c1 = 3, c2 = (a == 0 ? 5 : 0);
        for (const auto& [ctrl, val] : control_patterns(c1, c2)) {
            check_against_matrix<Prec, Space>(n, gate::Swap<Prec>(a, b, ctrl, val), "Swap");
            check_against_matrix<Prec, Space>(n, gate::Ecr<Prec>(a, b, ctrl, val), "Ecr");
        }
    }
}

TYPED_TEST(DMExtendedTest, DenseMatrixGates) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 6;
    std::mt19937_64 rng(42);
    const std::vector<std::vector<std::uint64_t>> target_sets = {
        {}, {2}, {5, 0}, {1, 4, 3}, {0, 2, 3, 5}, {0, 1, 2, 4, 5}};
    for (const auto& targets : target_sets) {
        std::vector<std::uint64_t> rest;
        for (std::uint64_t q = 0; q < n; ++q)
            if (std::ranges::find(targets, q) == targets.end()) rest.push_back(q);
        auto patterns = control_patterns(rest.empty() ? 0 : rest[0], rest.size() > 1 ? rest[1] : 0);
        if (rest.empty()) patterns.resize(1);
        if (rest.size() == 1) patterns.resize(3);
        for (const auto& [ctrl, val] : patterns) {
            const std::uint64_t k = targets.size();
            const std::string what =
                "k=" + std::to_string(k) + " #ctrl=" + std::to_string(ctrl.size());
            check_against_matrix<Prec, Space>(
                n,
                gate::DenseMatrix<Prec, Space>(targets, random_unitary(k, rng), ctrl, val),
                "Dense unitary " + what);
            check_against_matrix<Prec, Space>(
                n,
                gate::DenseMatrix<Prec, Space>(targets, random_matrix(k, rng), ctrl, val),
                "Dense general " + what);
        }
    }
}

TYPED_TEST(DMExtendedTest, PauliGatesAllControlPatterns) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 6;
    for (const std::string pauli : {"Z 0 Z 4", "X 1", "X 0 Y 2 Z 5", "Y 5 Y 0", "I 2"}) {
        for (const auto& [ctrl, val] : control_patterns(3, 1)) {
            if (pauli.find('1') != std::string::npos && !ctrl.empty()) continue;
            PauliOperator<Prec> op(pauli, StdComplex(0.6, 0.8));
            check_against_matrix<Prec, Space>(
                n, gate::Pauli<Prec>(op, ctrl, val), "Pauli " + pauli);
            check_against_matrix<Prec, Space>(
                n,
                gate::PauliRotation<Prec>(PauliOperator<Prec>(pauli), 0.9, ctrl, val),
                "PauliRotation " + pauli);
        }
    }
}

TYPED_TEST(DMExtendedTest, MultiSwapPermutation) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 5;
    for (const std::vector<std::uint64_t>& dst :
         {std::vector<std::uint64_t>{1, 2, 3, 4, 0}, std::vector<std::uint64_t>{4, 3, 0, 2, 1}}) {
        auto dm = DensityMatrix<Prec, Space>::Haar_random_state(n);
        const ComplexMatrix rho = dm.get_matrix();
        gate::Permutation<Prec, Space>(dst)->update_quantum_state(dm);
        const ComplexMatrix p = get_eigen_matrix_full_qubit_reorder(dst);
        expect_matrix_near<Prec>(
            dm.get_matrix(), p * rho * p.adjoint(), eps<Prec> * 10, "Permutation");
    }
}
