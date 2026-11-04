// Density-matrix coverage for every gate kind, every control pattern and the gates that
// previously had no DensityMatrix support (SparseMatrix, Probabilistic, Measurement,
// ParamGate, Circuit).

#include <gtest/gtest.h>

#include <Eigen/QR>
#include <random>
#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>
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

TYPED_TEST(DMExtendedTest, DenseAndSparseMatrixGates) {
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
            if (k == 0) continue;
            ComplexMatrix sparse_src = random_matrix(k, rng);
            for (Eigen::Index i = 0; i < sparse_src.rows(); ++i)
                for (Eigen::Index j = 0; j < sparse_src.cols(); ++j)
                    if ((i + 2 * j) % 3 == 0) sparse_src(i, j) = 0;
            check_against_matrix<Prec, Space>(
                n,
                gate::SparseMatrix<Prec, Space>(targets, sparse_src.sparseView(), ctrl, val),
                "Sparse " + what);
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

namespace {
// Σ p_k U_k ρ U_k† reference.
template <Precision Prec, ExecutionSpace Space>
void check_mixture(std::uint64_t n,
                   const Gate<Prec>& gate,
                   const std::vector<double>& probs,
                   const std::vector<Gate<Prec>>& gates,
                   const std::string& what) {
    auto dm = DensityMatrix<Prec, Space>::Haar_random_state(n);
    const ComplexMatrix rho = dm.get_matrix();
    gate->update_quantum_state(dm);
    ComplexMatrix expected = ComplexMatrix::Zero(rho.rows(), rho.cols());
    for (std::size_t i = 0; i < probs.size(); ++i) {
        const ComplexMatrix u = full_matrix(gates[i], n);
        expected += probs[i] * u * rho * u.adjoint();
    }
    expect_matrix_near<Prec>(dm.get_matrix(), expected, eps<Prec> * 10, what);
}
}  // namespace

TYPED_TEST(DMExtendedTest, ProbabilisticGateIsAppliedAsMixture) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 5;
    std::mt19937_64 rng(7);

    // Single-qubit Pauli channels (superoperator fast path).
    {
        const double p = 0.12;
        check_mixture<Prec, Space>(
            n,
            gate::DepolarizingNoise<Prec>(3, p),
            {p / 3, p / 3, p / 3, 1 - p},
            {gate::X<Prec>(3), gate::Y<Prec>(3), gate::Z<Prec>(3), gate::I<Prec>()},
            "Depolarizing");
        check_mixture<Prec, Space>(n,
                                   gate::BitFlipNoise<Prec>(0, p),
                                   {p, 1 - p},
                                   {gate::X<Prec>(0), gate::I<Prec>()},
                                   "BitFlip");
    }
    // Two-qubit union with controls and a global phase (fast path).
    {
        std::vector<Gate<Prec>> gates = {
            gate::CX<Prec>(1, 4),
            gate::H<Prec>(4, {1}, {0}),
            gate::GlobalPhase<Prec>(0.3, {4}),
            gate::DenseMatrix<Prec, Space>({1, 4}, random_unitary(2, rng))};
        std::vector<double> probs = {0.1, 0.2, 0.3, 0.4};
        check_mixture<Prec, Space>(
            n, gate::Probabilistic<Prec>(probs, gates), probs, gates, "2q mixture");
    }
    // Union over more than two qubits and a zero-probability term (general path).
    {
        std::vector<Gate<Prec>> gates = {
            gate::CCX<Prec>(0, 2, 4),
            gate::Swap<Prec>(1, 3),
            gate::RZ<Prec>(2, 0.5),
            gate::DenseMatrix<Prec, Space>({0, 1, 3}, random_unitary(3, rng))};
        std::vector<double> probs = {0.25, 0., 0.35, 0.4};
        check_mixture<Prec, Space>(
            n, gate::Probabilistic<Prec>(probs, gates), probs, gates, "5q mixture");
    }
    // 3-qubit union with identity / global-phase terms folded into a scalar (Kraus path).
    {
        std::vector<Gate<Prec>> gates = {
            gate::DenseMatrix<Prec, Space>({0, 2, 3}, random_unitary(3, rng)),
            gate::I<Prec>(),
            gate::GlobalPhase<Prec>(0.7),
            gate::X<Prec>(2, {0, 3}, {1, 0})};
        std::vector<double> probs = {0.3, 0.4, 0.1, 0.2};
        check_mixture<Prec, Space>(
            n, gate::Probabilistic<Prec>(probs, gates), probs, gates, "3q mixture");
    }
    // Union over more than six qubits (general path).
    {
        constexpr std::uint64_t n7 = 7;
        std::vector<Gate<Prec>> gates = {
            gate::CCX<Prec>(0, 2, 6),
            gate::DenseMatrix<Prec, Space>({1, 3, 5}, random_unitary(3, rng)),
            gate::Swap<Prec>(4, 6)};
        std::vector<double> probs = {0.25, 0.35, 0.4};
        check_mixture<Prec, Space>(
            n7, gate::Probabilistic<Prec>(probs, gates), probs, gates, "7q mixture");
    }
    // Nested probabilistic gates are flattened.
    {
        auto inner = gate::Probabilistic<Prec>({0.5, 0.5}, {gate::X<Prec>(1), gate::Z<Prec>(2)});
        check_mixture<Prec, Space>(n,
                                   gate::Probabilistic<Prec>({0.4, 0.6}, {inner, gate::H<Prec>(0)}),
                                   {0.2, 0.2, 0.6},
                                   {gate::X<Prec>(1), gate::Z<Prec>(2), gate::H<Prec>(0)},
                                   "nested");
    }
}

TYPED_TEST(DMExtendedTest, MeasurementProjectsAndRecords) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 4;
    for (bool reset : {false, true}) {
        for (std::uint64_t seed = 0; seed < 8; ++seed) {
            const std::uint64_t target = seed % n;
            auto dm = DensityMatrix<Prec, Space>::Haar_random_state(n, seed);
            const ComplexMatrix rho = dm.get_matrix();
            ClassicalRegister reg(2);
            gate::Measurement<Prec>(target, 1, reset)->update_quantum_state(dm, reg, seed);
            const bool one = reg[1];

            const ComplexMatrix proj =
                get_expanded_eigen_matrix_with_identity(target, one ? make_P1() : make_P0(), n);
            ComplexMatrix expected = proj * rho * proj.adjoint();
            const double p = expected.trace().real();
            ASSERT_GT(p, 0.);
            expected /= p;
            if (reset && one) {
                const ComplexMatrix x =
                    get_expanded_eigen_matrix_with_identity(target, make_X(), n);
                expected = x * expected * x.adjoint();
            }
            expect_matrix_near<Prec>(dm.get_matrix(), expected, eps<Prec> * 10, "Measurement");
        }
    }
}

TYPED_TEST(DMExtendedTest, ParamGatesMatchFixedGates) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 5;
    const double coef = 1.5, param = 0.37;
    const std::vector<std::pair<ParamGate<Prec>, Gate<Prec>>> cases = {
        {gate::ParamRX<Prec>(2, coef), gate::RX<Prec>(2, coef * param)},
        {gate::ParamRY<Prec>(0, coef, {3}, {0}), gate::RY<Prec>(0, coef * param, {3}, {0})},
        {gate::ParamRZ<Prec>(4, coef, {1, 2}), gate::RZ<Prec>(4, coef * param, {1, 2})},
        {gate::ParamPauliRotation<Prec>(PauliOperator<Prec>("X 0 Y 3"), coef, {2}),
         gate::PauliRotation<Prec>(PauliOperator<Prec>("X 0 Y 3"), coef * param, {2})},
    };
    for (const auto& [pg, g] : cases) {
        auto dm1 = DensityMatrix<Prec, Space>::Haar_random_state(n);
        auto dm2 = dm1.copy();
        pg->update_quantum_state(dm1, param);
        g->update_quantum_state(dm2);
        expect_matrix_near<Prec>(dm1.get_matrix(), dm2.get_matrix(), eps<Prec> * 10, "ParamGate");
    }

    // ParamProbabilistic mixes fixed and parametric gates.
    auto pp = gate::ParamProbabilistic<Prec>({0.3, 0.7},
                                             {gate::X<Prec>(1), gate::ParamRY<Prec>(1, coef)});
    check_mixture<Prec, Space>(
        n,
        gate::Probabilistic<Prec>({0.3, 0.7}, {gate::X<Prec>(1), gate::RY<Prec>(1, coef * param)}),
        {0.3, 0.7},
        {gate::X<Prec>(1), gate::RY<Prec>(1, coef * param)},
        "reference mixture");
    auto dm1 = DensityMatrix<Prec, Space>::Haar_random_state(n);
    auto dm2 = dm1.copy();
    pp->update_quantum_state(dm1, param);
    gate::Probabilistic<Prec>({0.3, 0.7}, {gate::X<Prec>(1), gate::RY<Prec>(1, coef * param)})
        ->update_quantum_state(dm2);
    expect_matrix_near<Prec>(
        dm1.get_matrix(), dm2.get_matrix(), eps<Prec> * 10, "ParamProbabilistic");
}

TYPED_TEST(DMExtendedTest, CircuitOnPureStateMatchesStateVector) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 5;
    std::mt19937_64 rng(3);
    Circuit<Prec> circuit;
    circuit.add_gate(gate::H<Prec>(0));
    circuit.add_gate(gate::CX<Prec>(0, 3));
    circuit.add_param_gate(gate::ParamRX<Prec>(2, 1.), "a");
    circuit.add_gate(gate::DenseMatrix<Prec, Space>({1, 2, 4}, random_unitary(3, rng), {0}));
    circuit.add_gate(gate::SparseMatrix<Prec, Space>({3, 4}, random_unitary(2, rng).sparseView()));
    circuit.add_param_gate(gate::ParamPauliRotation<Prec>(PauliOperator<Prec>("Z 1 X 4")), "b");
    circuit.add_gate(gate::Swap<Prec>(1, 4, {2}, {0}));
    const std::map<std::string, double> params = {{"a", 0.3}, {"b", -0.8}};

    auto sv = StateVector<Prec, Space>::Haar_random_state(n);
    DensityMatrix<Prec, Space> dm(sv);
    circuit.update_quantum_state(sv, params);
    circuit.update_quantum_state(dm, params);
    expect_matrix_near<Prec>(
        dm.get_matrix(), DensityMatrix<Prec, Space>(sv).get_matrix(), eps<Prec> * 10, "Circuit");
}

TYPED_TEST(DMExtendedTest, CircuitWithNoiseAndMeasurement) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 3;
    Circuit<Prec> circuit;
    circuit.add_gate(gate::H<Prec>(0));
    circuit.add_gate(gate::DepolarizingNoise<Prec>(0, 0.2));
    circuit.add_gate(gate::CX<Prec>(0, 1));
    circuit.add_gate(gate::Measurement<Prec>(1, 0, true));

    DensityMatrix<Prec, Space> dm(n);
    ClassicalRegister reg(1);
    circuit.update_quantum_state(dm, reg, {}, 1234);
    ASSERT_NEAR(dm.get_trace().real(), 1., eps<Prec> * 10);
    // After measuring qubit 1 with reset, qubit 1 is |0> and qubit 0 collapsed to `reg[0]`.
    const ComplexMatrix m = dm.get_matrix();
    const std::uint64_t basis = reg[0] ? 0b001 : 0b000;
    ASSERT_NEAR(m(basis, basis).real(), 1., eps<Prec> * 10);

    // A circuit with a measurement needs a classical register.
    DensityMatrix<Prec, Space> dm2(n);
    ASSERT_THROW(circuit.update_quantum_state(dm2), std::runtime_error);
}
