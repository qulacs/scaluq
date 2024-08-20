#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <gate/gate.hpp>
#include <gate/gate_factory.hpp>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

const auto eps = 1e-12;

template <Gate (*QuantumGateConstructor)()>
void run_random_gate_apply(UINT n_qubits) {
    const int dim = 1ULL << n_qubits;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const Gate gate = QuantumGateConstructor();
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        test_state = test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

template <Gate (*QuantumGateConstructor)(double)>
void run_random_gate_apply(UINT n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = M_PI * random.uniform();
        const Gate gate = QuantumGateConstructor(angle);
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        test_state = std::polar(1., angle) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

template <Gate (*QuantumGateConstructor)(UINT)>
void run_random_gate_apply(UINT n_qubits, std::function<Eigen::MatrixXcd()> matrix_factory) {
    const auto matrix = matrix_factory();
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const UINT target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target);
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

template <Gate (*QuantumGateConstructor)(UINT, double)>
void run_random_gate_apply(UINT n_qubits, std::function<Eigen::MatrixXcd(double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = M_PI * random.uniform();
        const auto matrix = matrix_factory(angle);
        const UINT target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target, angle);
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_IBMQ(
    UINT n_qubits, std::function<Eigen::MatrixXcd(double, double, double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int gate_type = 0; gate_type < 3; gate_type++) {
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            double theta = M_PI * random.uniform();
            double phi = M_PI * random.uniform();
            double lambda = M_PI * random.uniform();
            if (gate_type == 0) {
                theta = 0;
                phi = 0;
            } else if (gate_type == 1) {
                theta = M_PI / 2;
            }
            const auto matrix = matrix_factory(theta, phi, lambda);
            const UINT target = random.int64() % n_qubits;
            Gate gate;
            if (gate_type == 0) {
                gate = gate::U1(target, lambda);
            } else if (gate_type == 1) {
                gate = gate::U2(target, phi, lambda);
            } else {
                gate = gate::U3(target, theta, phi, lambda);
            }
            gate->update_quantum_state(state);
            state_cp = state.amplitudes();

            test_state =
                get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_gate_apply_two_qubit(UINT n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    std::function<Eigen::MatrixXcd(UINT, UINT, UINT)> func_eig;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        for (int g = 0; g < 2; g++) {
            Gate gate;
            auto state_cp = state.amplitudes();
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            UINT target = random.int64() % n_qubits;
            UINT control = random.int64() % n_qubits;
            if (target == control) target = (target + 1) % n_qubits;
            if (g == 0) {
                gate = gate::CX(control, target);
                func_eig = get_eigen_matrix_full_qubit_CX;
            } else {
                gate = gate::CZ(control, target);
                func_eig = get_eigen_matrix_full_qubit_CZ;
            }
            gate->update_quantum_state(state);
            state_cp = state.amplitudes();

            Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
            test_state = test_mat * test_state;

            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }

    func_eig = get_eigen_matrix_full_qubit_Swap;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        UINT target = random.int64() % n_qubits;
        UINT control = random.int64() % n_qubits;
        if (target == control) target = (target + 1) % n_qubits;
        auto gate = gate::Swap(control, target);
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
        test_state = test_mat * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_fused(UINT n_qubits, UINT target0, UINT target1, UINT block_size) {
    const UINT dim = 1ULL << n_qubits;
    StateVector state_ref = StateVector::Haar_random_state(n_qubits);
    StateVector state = state_ref.copy();

    // update "state_ref" using Swap gate
    for (UINT i = 0; i < block_size; i++) {
        auto swap_gate = gate::Swap(target0 + i, target1 + i);
        swap_gate->update_quantum_state(state_ref);
    }
    auto state_ref_cp = state_ref.amplitudes();

    auto fused_swap_gate = gate::FusedSwap(target0, target1, block_size);
    fused_swap_gate->update_quantum_state(state);
    auto state_cp = state.amplitudes();

    for (UINT i = 0; i < dim; i++) {
        ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - (StdComplex)state_ref_cp[i]), 0, eps);
    }
}

void run_random_gate_apply_pauli(UINT n_qubits) {
    const UINT dim = 1ULL << n_qubits;
    Random random;
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    Eigen::MatrixXcd matrix;

    // Test for PauliGate
    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        auto state_bef = state.copy();

        for (UINT i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::vector<UINT> target_vec, pauli_id_vec;
        for (UINT target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

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

        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate pauli_gate = gate::Pauli(pauli);
        pauli_gate->update_quantum_state(state);

        state_cp = state.amplitudes();
        test_state = matrix * test_state;

        // check if the state is updated correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }

        auto state_bef_cp = state_bef.amplitudes();
        Gate pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(state);
        state_cp = state.amplitudes();

        // check if the state is restored correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)(state_cp[i] - state_bef_cp[i])), 0, eps);
        }
    }

    // Test for PauliRotationGate
    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        auto state_bef = state.copy();
        assert(test_state.size() == (int)state_cp.size());
        for (UINT i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        const double angle = M_PI * random.uniform();
        std::vector<UINT> target_vec, pauli_id_vec;
        for (UINT target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

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
        matrix = std::cos(angle / 2) * Eigen::MatrixXcd::Identity(dim, dim) -
                 Complex(0, 1) * std::sin(angle / 2) * matrix;
        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate pauli_gate = gate::PauliRotation(pauli, angle);
        pauli_gate->update_quantum_state(state);
        state_cp = state.amplitudes();
        test_state = matrix * test_state;
        assert((int)state_cp.size() == test_state.size());
        // check if the state is updated correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
        Gate pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(state);
        state_cp = state.amplitudes();
        auto state_bef_cp = state_bef.amplitudes();
        // check if the state is restored correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)(state_cp[i] - state_bef_cp[i])), 0, eps);
        }
    }
}

void run_random_gate_apply_single_dense(UINT n_qubits) {
    const UINT dim = 1ULL << n_qubits;
    const UINT max_repeat = 10;

    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U;
    UINT target;
    Kokkos::View<Complex**> mat_view("mat_view", 2, 2);
    Random random;
    for (UINT rep = 0; rep < max_repeat; rep++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
        for (UINT i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        target = random.int64() % n_qubits;
        std::vector<UINT> target_list = {target};
        U = get_eigen_matrix_random_single_qubit_unitary();
        ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                mat(i, j) = U(i, j);
            }
        }
        std::vector<UINT> control_list = {};
        Gate dense_gate = gate::DenseMatrix(mat, target_list, control_list);
        dense_gate->update_quantum_state(state);
        test_state = get_expanded_eigen_matrix_with_identity(target, U, n_qubits) * test_state;
        state_cp = state.amplitudes();
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_sparse(UINT n_qubits) {
    const UINT dim = 1ULL << n_qubits;
    const UINT max_repeat = 10;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    SparseComplexMatrix mat;
    std::vector<UINT> targets(3);
    std::vector<UINT> index_list;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> u1, u2, u3;
    Eigen::Matrix<StdComplex, 8, 8, Eigen::RowMajor> Umerge;
    for (UINT i = 0; i < n_qubits; i++) {
        index_list.push_back(i);
    }
    for (UINT rep = 0; rep < max_repeat; rep++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (UINT i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        std::shuffle(index_list.begin(), index_list.end(), engine);
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        targets[2] = index_list[2];
        u1 = make_sparse_complex_matrix(1, 0.2);
        u2 = make_sparse_complex_matrix(1, 0.2);
        u3 = make_sparse_complex_matrix(1, 0.2);
        Umerge = internal::kronecker_product(u3, internal::kronecker_product(u2, u1));
        test_state = get_expanded_eigen_matrix_with_identity(targets[2], u3, n_qubits) *
                     get_expanded_eigen_matrix_with_identity(targets[1], u2, n_qubits) *
                     get_expanded_eigen_matrix_with_identity(targets[0], u1, n_qubits) * test_state;

        std::vector<UINT> target_list = {targets[0], targets[1], targets[2]};
        std::vector<UINT> control_list = {};
        // ComplexMatrix mat(8, 8);
        // for (int i = 0; i < 8; i++) {
        //     for (int j = 0; j < 8; j++) {
        //         mat(i, j) = Umerge(i, j);
        //     }
        // }
        // Gate dense_gate = gate::DenseMatrix(mat, target_list, control_list);
        // dense_gate->update_quantum_state(state);

        mat = Umerge.sparseView();
        Gate sparse_gate = gate::SparseMatrix(mat, target_list, control_list);
        sparse_gate->update_quantum_state(state);
        state_cp = state.amplitudes();
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_general_dense(UINT n_qubits) {
    const UINT dim = 1ULL << n_qubits;
    const UINT max_repeat = 10;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U1, U2, U3;
    std::vector<UINT> targets(3);
    std::vector<UINT> index_list;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT i = 0; i < n_qubits; i++) {
        index_list.push_back(i);
    }
    // general single
    {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> Umerge;
        for (UINT rep = 0; rep < max_repeat; rep++) {
            StateVector state = StateVector::Haar_random_state(n_qubits);
            auto state_cp = state.amplitudes();
            for (UINT i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_single_qubit_unitary();
            ComplexMatrix mat(U1.rows(), U1.cols());
            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            Umerge = U1;
            test_state =
                get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) * test_state;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    mat(i, j) = U1(i, j);
                }
            }
            std::vector<UINT> target_list = {targets[0]};
            std::vector<UINT> control_list = {};
            Gate dense_gate = gate::DenseMatrix(mat, target_list, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.amplitudes();
            for (UINT i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }
    // general double
    {
        Eigen::Matrix<StdComplex, 4, 4, Eigen::RowMajor> Umerge;
        for (UINT rep = 0; rep < max_repeat; rep++) {
            StateVector state = StateVector::Haar_random_state(n_qubits);
            auto state_cp = state.amplitudes();
            for (UINT i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_single_qubit_unitary();
            U2 = get_eigen_matrix_random_single_qubit_unitary();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            Umerge = internal::kronecker_product(U2, U1);
            ComplexMatrix mat(Umerge.rows(), Umerge.cols());
            test_state = get_expanded_eigen_matrix_with_identity(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) *
                         test_state;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    mat(i, j) = Umerge(i, j);
                }
            }
            std::vector<UINT> target_list = {targets[0], targets[1]};
            std::vector<UINT> control_list = {};
            Gate dense_gate = gate::DenseMatrix(mat, target_list, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.amplitudes();
            for (UINT i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }
    // general triple
    {
        Eigen::Matrix<StdComplex, 8, 8, Eigen::RowMajor> Umerge;
        for (UINT rep = 0; rep < max_repeat; rep++) {
            StateVector state = StateVector::Haar_random_state(n_qubits);
            auto state_cp = state.amplitudes();
            for (UINT i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_single_qubit_unitary();
            U2 = get_eigen_matrix_random_single_qubit_unitary();
            U3 = get_eigen_matrix_random_single_qubit_unitary();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            targets[2] = index_list[2];
            Umerge = internal::kronecker_product(U3, internal::kronecker_product(U2, U1));
            ComplexMatrix mat(Umerge.rows(), Umerge.cols());

            test_state = get_expanded_eigen_matrix_with_identity(targets[2], U3, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) *
                         test_state;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    mat(i, j) = Umerge(i, j);
                }
            }
            std::vector<UINT> target_list = {targets[0], targets[1], targets[2]};
            std::vector<UINT> control_list = {};
            Gate dense_gate = gate::DenseMatrix(mat, target_list, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.amplitudes();
            for (UINT i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }
}

TEST(GateTest, ApplyI) { run_random_gate_apply<gate::I>(5); }
TEST(GateTest, ApplyGlobalPhase) { run_random_gate_apply<gate::GlobalPhase>(5); }
TEST(GateTest, ApplyX) { run_random_gate_apply<gate::X>(5, make_X); }
TEST(GateTest, ApplyY) { run_random_gate_apply<gate::Y>(5, make_Y); }
TEST(GateTest, ApplyZ) { run_random_gate_apply<gate::Z>(5, make_Z); }
TEST(GateTest, ApplyH) { run_random_gate_apply<gate::H>(5, make_H); }
TEST(GateTest, ApplyS) { run_random_gate_apply<gate::S>(5, make_S); }
TEST(GateTest, ApplySdag) { run_random_gate_apply<gate::Sdag>(5, make_Sdag); }
TEST(GateTest, ApplyT) { run_random_gate_apply<gate::T>(5, make_T); }
TEST(GateTest, ApplyTdag) { run_random_gate_apply<gate::Tdag>(5, make_Tdag); }
TEST(GateTest, ApplySqrtX) { run_random_gate_apply<gate::SqrtX>(5, make_SqrtX); }
TEST(GateTest, ApplySqrtY) { run_random_gate_apply<gate::SqrtY>(5, make_SqrtY); }
TEST(GateTest, ApplySqrtXdag) { run_random_gate_apply<gate::SqrtXdag>(5, make_SqrtXdag); }
TEST(GateTest, ApplySqrtYdag) { run_random_gate_apply<gate::SqrtYdag>(5, make_SqrtYdag); }
TEST(GateTest, ApplyP0) { run_random_gate_apply<gate::P0>(5, make_P0); }
TEST(GateTest, ApplyP1) { run_random_gate_apply<gate::P1>(5, make_P1); }
TEST(GateTest, ApplyRX) { run_random_gate_apply<gate::RX>(5, make_RX); }
TEST(GateTest, ApplyRY) { run_random_gate_apply<gate::RY>(5, make_RY); }
TEST(GateTest, ApplyRZ) { run_random_gate_apply<gate::RZ>(5, make_RZ); }

TEST(GateTest, ApplyIBMQ) { run_random_gate_apply_IBMQ(5, make_U); }

TEST(GateTest, ApplyTwoQubit) { run_random_gate_apply_two_qubit(5); }
TEST(GateTest, ApplyFused) {
    UINT n_qubits = 10;
    for (UINT t0 = 0; t0 < n_qubits; t0++) {
        for (UINT t1 = 0; t1 < n_qubits; t1++) {
            if (t0 == t1) continue;
            UINT max_bs =
                std::min((t0 < t1) ? (t1 - t0) : (t0 - t1), std::min(n_qubits - t0, n_qubits - t1));
            for (UINT bs = 1; bs <= max_bs; bs++) {
                run_random_gate_apply_fused(n_qubits, t0, t1, bs);
            }
        }
    }
}

TEST(GateTest, ApplyPauliGate) { run_random_gate_apply_pauli(5); }
TEST(GateTest, ApplySparseMatrixGate) { run_random_gate_apply_sparse(6); }
TEST(GateTest, ApplyDenseMatrixGate) {
    run_random_gate_apply_single_dense(6);
    run_random_gate_apply_general_dense(6);
}

TEST(GateTest, ApplyProbablisticGate) {
    auto probgate = gate::Probablistic({.1, .9}, {gate::X(0), gate::I()});
    UINT x_cnt = 0, i_cnt = 0;
    StateVector state(1);
    for ([[maybe_unused]] auto _ : std::views::iota(0, 100)) {
        UINT before = state.sampling(1)[0];
        probgate->update_quantum_state(state);
        UINT after = state.sampling(1)[0];
        if (before != after) {
            x_cnt++;
        } else {
            i_cnt++;
        }
    }
    // These test is probablistic, but pass at least 99.99% cases.
    ASSERT_GT(x_cnt, 0);
    ASSERT_GT(i_cnt, 0);
    ASSERT_LT(x_cnt, i_cnt);
}
