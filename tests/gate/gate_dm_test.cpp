#include <gtest/gtest.h>

#include <Eigen/Core>
#include <numbers>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/state/density_matrix.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class DMGateTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(DMGateTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space>
void run_dm_gate_apply(std::uint64_t n_qubits, const Gate<Prec>& gate, const ComplexMatrix& U) {
    auto dm = DensityMatrix<Prec, Space>::Haar_random_state(n_qubits);
    ComplexMatrix rho = dm.get_matrix();

    gate->update_quantum_state(dm);
    ComplexMatrix rho_after = dm.get_matrix();

    ComplexMatrix expected = U * rho * U.adjoint();

    std::uint64_t dim = 1ULL << n_qubits;
    for (std::uint64_t i = 0; i < dim; i++) {
        for (std::uint64_t j = 0; j < dim; j++) {
            ASSERT_NEAR(std::abs(rho_after(i, j) - expected(i, j)), 0., eps<Prec>)
                << "at (" << i << ", " << j << ")";
        }
    }
}

template <Precision Prec, ExecutionSpace Space>
void run_dm_single_qubit_gate(std::uint64_t n_qubits,
                              std::uint64_t target,
                              const Gate<Prec>& gate,
                              const ComplexMatrix& gate_matrix) {
    ComplexMatrix U =
        get_expanded_eigen_matrix_with_identity(target, gate_matrix, n_qubits);
    run_dm_gate_apply<Prec, Space>(n_qubits, gate, U);
}

template <Precision Prec, ExecutionSpace Space>
void run_dm_gate_apply_against_state_vector(std::uint64_t n_qubits, const Gate<Prec>& gate) {
    constexpr double weight1 = 0.4;
    constexpr double weight2 = 0.6;
    auto state1 = StateVector<Prec, Space>::Haar_random_state(n_qubits);
    auto state2 = StateVector<Prec, Space>::Haar_random_state(n_qubits);
    DensityMatrix<Prec, Space> dm1(state1);
    DensityMatrix<Prec, Space> dm2(state2);
    dm1.multiply_coef(weight1);
    dm1.add_density_matrix_with_coef(weight2, dm2);

    gate->update_quantum_state(dm1);
    gate->update_quantum_state(state1);
    gate->update_quantum_state(state2);

    DensityMatrix<Prec, Space> expected1(state1);
    DensityMatrix<Prec, Space> expected2(state2);
    expected1.multiply_coef(weight1);
    expected1.add_density_matrix_with_coef(weight2, expected2);
    const auto actual = dm1.get_matrix();
    const auto expected = expected1.get_matrix();
    for (std::uint64_t row = 0; row < dm1.dim(); ++row) {
        for (std::uint64_t col = 0; col < dm1.dim(); ++col) {
            ASSERT_NEAR(std::abs(actual(row, col) - expected(row, col)), 0., eps<Prec> * 10)
                << "at (" << row << ", " << col << ")";
        }
    }
}

TYPED_TEST(DMGateTest, ApplyI) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;
    for (int rep = 0; rep < 5; rep++) {
        auto gate_obj = gate::I<Prec>();
        ComplexMatrix U = ComplexMatrix::Identity(dim, dim);
        run_dm_gate_apply<Prec, Space>(n, gate_obj, U);
    }
}

TYPED_TEST(DMGateTest, ApplyGlobalPhase) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        double angle = random.uniform() * 2 * std::numbers::pi;
        auto gate = gate::GlobalPhase<Prec>(angle, {}, {});
        // global phase: e^{i*angle} * Identity
        ComplexMatrix U = std::exp(StdComplex(0, angle)) * ComplexMatrix::Identity(dim, dim);
        run_dm_gate_apply<Prec, Space>(n, gate, U);
    }
}

TYPED_TEST(DMGateTest, ApplyX) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(n, target, gate::X<Prec>(target, {}, {}), make_X());
    }
}

TYPED_TEST(DMGateTest, ApplyY) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(n, target, gate::Y<Prec>(target, {}, {}), make_Y());
    }
}

TYPED_TEST(DMGateTest, ApplyZ) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(n, target, gate::Z<Prec>(target, {}, {}), make_Z());
    }
}

TYPED_TEST(DMGateTest, ApplyH) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(n, target, gate::H<Prec>(target, {}, {}), make_H());
    }
}

TYPED_TEST(DMGateTest, ApplyS) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(n, target, gate::S<Prec>(target, {}, {}), make_S());
    }
}

TYPED_TEST(DMGateTest, ApplySdag) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::Sdag<Prec>(target, {}, {}), make_Sdag());
    }
}

TYPED_TEST(DMGateTest, ApplyT) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(n, target, gate::T<Prec>(target, {}, {}), make_T());
    }
}

TYPED_TEST(DMGateTest, ApplyTdag) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::Tdag<Prec>(target, {}, {}), make_Tdag());
    }
}

TYPED_TEST(DMGateTest, ApplySqrtX) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::SqrtX<Prec>(target, {}, {}), make_SqrtX());
    }
}

TYPED_TEST(DMGateTest, ApplySqrtXdag) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::SqrtXdag<Prec>(target, {}, {}), make_SqrtXdag());
    }
}

TYPED_TEST(DMGateTest, ApplySqrtY) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::SqrtY<Prec>(target, {}, {}), make_SqrtY());
    }
}

TYPED_TEST(DMGateTest, ApplySqrtYdag) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::SqrtYdag<Prec>(target, {}, {}), make_SqrtYdag());
    }
}

TYPED_TEST(DMGateTest, ApplyP0) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::P0<Prec>(target, {}, {}), make_P0());
    }
}

TYPED_TEST(DMGateTest, ApplyP1) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::P1<Prec>(target, {}, {}), make_P1());
    }
}

TYPED_TEST(DMGateTest, ApplyRX) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        double angle = random.uniform() * 2 * std::numbers::pi;
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::RX<Prec>(target, angle, {}, {}), make_RX(angle));
    }
}

TYPED_TEST(DMGateTest, ApplyRY) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        double angle = random.uniform() * 2 * std::numbers::pi;
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::RY<Prec>(target, angle, {}, {}), make_RY(angle));
    }
}

TYPED_TEST(DMGateTest, ApplyRZ) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        double angle = random.uniform() * 2 * std::numbers::pi;
        std::uint64_t target = random.int64() % n;
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::RZ<Prec>(target, angle, {}, {}), make_RZ(angle));
    }
}

TYPED_TEST(DMGateTest, ApplyU1) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        double lambda = random.uniform() * 2 * std::numbers::pi;
        std::uint64_t target = random.int64() % n;
        ComplexMatrix gate_mat = make_U(0, 0, lambda);
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::U1<Prec>(target, lambda, {}, {}), gate_mat);
    }
}

TYPED_TEST(DMGateTest, ApplyU2) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        double phi = random.uniform() * 2 * std::numbers::pi;
        double lambda = random.uniform() * 2 * std::numbers::pi;
        std::uint64_t target = random.int64() % n;
        ComplexMatrix gate_mat = make_U(std::numbers::pi / 2, phi, lambda);
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::U2<Prec>(target, phi, lambda, {}, {}), gate_mat);
    }
}

TYPED_TEST(DMGateTest, ApplyU3) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        double theta = random.uniform() * std::numbers::pi;
        double phi = random.uniform() * 2 * std::numbers::pi;
        double lambda = random.uniform() * 2 * std::numbers::pi;
        std::uint64_t target = random.int64() % n;
        ComplexMatrix gate_mat = make_U(theta, phi, lambda);
        run_dm_single_qubit_gate<Prec, Space>(
            n, target, gate::U3<Prec>(target, theta, phi, lambda, {}, {}), gate_mat);
    }
}

TYPED_TEST(DMGateTest, ApplySwap) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t t1 = random.int64() % n;
        std::uint64_t t2;
        do {
            t2 = random.int64() % n;
        } while (t2 == t1);
        auto gate_obj = gate::Swap<Prec>(t1, t2, {}, {});
        ComplexMatrix U = get_eigen_matrix_full_qubit_Swap(t1, t2, n);
        run_dm_gate_apply<Prec, Space>(n, gate_obj, U);
    }
}

TYPED_TEST(DMGateTest, ApplyEcr) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t ctrl = random.int64() % n;
        std::uint64_t tgt;
        do {
            tgt = random.int64() % n;
        } while (tgt == ctrl);
        auto gate_obj = gate::Ecr<Prec>(ctrl, tgt, {}, {});
        ComplexMatrix U = get_eigen_matrix_full_qubit_Ecr(ctrl, tgt, n);
        run_dm_gate_apply<Prec, Space>(n, gate_obj, U);
    }
}

TYPED_TEST(DMGateTest, ApplyControlledX) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t ctrl = random.int64() % n;
        std::uint64_t tgt;
        do {
            tgt = random.int64() % n;
        } while (tgt == ctrl);
        auto gate_obj = gate::X<Prec>(tgt, {ctrl}, {1});
        ComplexMatrix U = get_eigen_matrix_full_qubit_CX(ctrl, tgt, n);
        run_dm_gate_apply<Prec, Space>(n, gate_obj, U);
    }
}

TYPED_TEST(DMGateTest, ApplyControlledZ) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t ctrl = random.int64() % n;
        std::uint64_t tgt;
        do {
            tgt = random.int64() % n;
        } while (tgt == ctrl);
        auto gate_obj = gate::Z<Prec>(tgt, {ctrl}, {1});
        ComplexMatrix U = get_eigen_matrix_full_qubit_CZ(ctrl, tgt, n);
        run_dm_gate_apply<Prec, Space>(n, gate_obj, U);
    }
}

TYPED_TEST(DMGateTest, ApplyControlledEcr) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n = 4;
    Random random;
    for (int rep = 0; rep < 5; rep++) {
        std::uint64_t ext_ctrl, phys_ctrl, phys_tgt;
        ext_ctrl = random.int64() % n;
        do {
            phys_ctrl = random.int64() % n;
        } while (phys_ctrl == ext_ctrl);
        do {
            phys_tgt = random.int64() % n;
        } while (phys_tgt == ext_ctrl || phys_tgt == phys_ctrl);
        auto gate_obj = gate::Ecr<Prec>(phys_ctrl, phys_tgt, {ext_ctrl}, {1});
        ComplexMatrix U = get_eigen_matrix_full_qubit_controlled_Ecr(ext_ctrl, phys_ctrl, phys_tgt, n);
        run_dm_gate_apply<Prec, Space>(n, gate_obj, U);
    }
}

TYPED_TEST(DMGateTest, ApplyDenseMatrix) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 5;

    ComplexMatrix scalar(1, 1);
    scalar << StdComplex(0.3, -0.2);
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::DenseMatrix<Prec, Space>({}, scalar));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::DenseMatrix<Prec, Space>({}, scalar, {1, 4}, {0, 1}));

    ComplexMatrix one_target(2, 2);
    one_target << StdComplex(0.2, 0.1), StdComplex(-0.3, 0.4), StdComplex(0.5, -0.2),
        StdComplex(0.1, 0.6);
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::DenseMatrix<Prec, Space>({2}, one_target));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::DenseMatrix<Prec, Space>({2}, one_target, {0, 4}, {1, 0}));

    ComplexMatrix two_target = ComplexMatrix::Zero(4, 4);
    two_target << 1., 0., 0., StdComplex(0., 0.5), 0., StdComplex(0.2, -0.1), 0.3, 0., 0., -0.4,
        StdComplex(0.1, 0.2), 0., StdComplex(-0.2, 0.3), 0., 0., 0.7;
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::DenseMatrix<Prec, Space>({1, 3}, two_target));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::DenseMatrix<Prec, Space>({3, 1}, two_target, {4}, {1}));

    ComplexMatrix three_target = ComplexMatrix::Zero(8, 8);
    for (std::uint64_t row = 0; row < 8; ++row) {
        three_target(row, row) = StdComplex(0.1 * (row + 1), -0.03 * row);
        three_target(row, (row + 3) % 8) = StdComplex(-0.02 * row, 0.05 * (row + 1));
    }
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::DenseMatrix<Prec, Space>({4, 0, 2}, three_target, {1}, {0}));
}

TYPED_TEST(DMGateTest, ApplyPauli) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 4;

    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::Pauli<Prec>(PauliOperator<Prec>("Z 0 Z 2")));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::Pauli<Prec>(PauliOperator<Prec>("X 0 Y 2 Z 3", StdComplex(0.6, 0.8))));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::Pauli<Prec>(PauliOperator<Prec>("X 0 Y 2"), {1}, {0}));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::Pauli<Prec>(PauliOperator<Prec>("Y 0 X 2"), {1, 3}, {0, 1}));
}

TYPED_TEST(DMGateTest, ApplyPauliRotation) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t n = 4;
    constexpr double angle = 0.37;

    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::PauliRotation<Prec>(PauliOperator<Prec>("Z 0 Z 2"), angle));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::PauliRotation<Prec>(PauliOperator<Prec>("X 0 Y 2 Z 3", 0.7), angle));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n, gate::PauliRotation<Prec>(PauliOperator<Prec>("X 0 Y 2"), angle, {1}, {0}));
    run_dm_gate_apply_against_state_vector<Prec, Space>(
        n,
        gate::PauliRotation<Prec>(
            PauliOperator<Prec>("Y 0 X 2"), angle, {1, 3}, {0, 1}));
}
