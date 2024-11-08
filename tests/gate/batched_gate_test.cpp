#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <gate/gate.hpp>
#include <gate/gate_factory.hpp>
#include <numbers>
#include <state/state_vector.hpp>
#include <state/state_vector_batched.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

const auto eps = 1e-12;
const std::uint64_t BATCH_SIZE = 10;
using StdComplex = std::complex<double>;

template <Gate (*QuantumGateConstructor)()>
void run_random_batched_gate_apply(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const Gate gate = QuantumGateConstructor();
        gate->update_quantum_state(states);
        auto states_cp = states.get_amplitudes();

        test_state = test_state;

        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

template <Gate (*QuantumGateConstructor)(double, const std::vector<std::uint64_t>&)>
void run_random_batched_gate_apply(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = std::numbers::pi * random.uniform();
        const Gate gate = QuantumGateConstructor(angle, {});
        gate->update_quantum_state(states);

        test_state = std::polar(1., angle) * test_state;

        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            auto states_cp = states.get_amplitudes();
            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

template <Gate (*QuantumGateConstructor)(std::uint64_t, const std::vector<std::uint64_t>&)>
void run_random_batched_gate_apply(std::uint64_t n_qubits,
                                   std::function<Eigen::MatrixXcd()> matrix_factory) {
    const auto matrix = matrix_factory();
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const std::uint64_t target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target, {});
        gate->update_quantum_state(states);

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

template <Gate (*QuantumGateConstructor)(std::uint64_t, double, const std::vector<std::uint64_t>&)>
void run_random_batched_gate_apply(std::uint64_t n_qubits,
                                   std::function<Eigen::MatrixXcd(double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = std::numbers::pi * random.uniform();
        const auto matrix = matrix_factory(angle);
        const std::uint64_t target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target, angle, {});
        gate->update_quantum_state(states);

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_batched_gate_apply_IBMQ(
    std::uint64_t n_qubits,
    std::function<Eigen::MatrixXcd(double, double, double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        for (int gate_type = 0; gate_type < 3; gate_type++) {
            auto state_cp = states.get_state_vector_at(0).get_amplitudes();
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            double theta = std::numbers::pi * random.uniform();
            double phi = std::numbers::pi * random.uniform();
            double lambda = std::numbers::pi * random.uniform();
            if (gate_type == 0) {
                theta = 0;
                phi = 0;
            } else if (gate_type == 1) {
                theta = std::numbers::pi / 2;
            }
            const auto matrix = matrix_factory(theta, phi, lambda);
            const std::uint64_t target = random.int64() % n_qubits;
            Gate gate;
            if (gate_type == 0) {
                gate = gate::U1(target, lambda, {});
            } else if (gate_type == 1) {
                gate = gate::U2(target, phi, lambda, {});
            } else {
                gate = gate::U3(target, theta, phi, lambda, {});
            }
            gate->update_quantum_state(states);

            test_state =
                get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

            auto states_cp = states.get_amplitudes();
            for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
                for (int i = 0; i < dim; i++) {
                    ASSERT_NEAR(
                        std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
                }
            }
        }
    }
}

void run_random_batched_gate_apply_two_target(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    std::function<Eigen::MatrixXcd(std::uint64_t, std::uint64_t, std::uint64_t)> func_eig;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        for (int g = 0; g < 2; g++) {
            Gate gate;
            auto state_cp = states.get_state_vector_at(0).get_amplitudes();
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            std::uint64_t target = random.int64() % n_qubits;
            std::uint64_t control = random.int64() % n_qubits;
            if (target == control) target = (target + 1) % n_qubits;
            if (g == 0) {
                gate = gate::CX(control, target);
                func_eig = get_eigen_matrix_full_qubit_CX;
            } else {
                gate = gate::CZ(control, target);
                func_eig = get_eigen_matrix_full_qubit_CZ;
            }
            gate->update_quantum_state(states);

            Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
            test_state = test_mat * test_state;

            auto states_cp = states.get_amplitudes();
            for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
                for (int i = 0; i < dim; i++) {
                    ASSERT_NEAR(
                        std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
                }
            }
        }
    }

    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::uint64_t target1 = random.int64() % n_qubits;
        std::uint64_t target2 = random.int64() % n_qubits;
        if (target1 == target2) target1 = (target1 + 1) % n_qubits;
        auto gate = gate::Swap(target1, target2);
        gate->update_quantum_state(states);

        Eigen::MatrixXcd test_mat = get_eigen_matrix_full_qubit_Swap(target1, target2, n_qubits);
        test_state = test_mat * test_state;

        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_batched_gate_apply_pauli(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    Eigen::MatrixXcd matrix;

    // Test for PauliGate
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        auto states_bef = states.copy();

        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
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
        pauli_gate->update_quantum_state(states);

        test_state = matrix * test_state;

        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (std::uint64_t i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }

        auto states_bef_cp = states_bef.get_amplitudes();
        Gate pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(states);
        states_cp = states.get_amplitudes();

        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (std::uint64_t i = 0; i < dim; i++) {
                ASSERT_NEAR(
                    std::abs((StdComplex)(states_cp[batch_id][i] - states_bef_cp[batch_id][i])),
                    0,
                    eps);
            }
        }
    }

    // Test for PauliRotationGate
    for (int repeat = 0; repeat < 10; repeat++) {
        auto states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        auto states_bef = states.copy();
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = std::numbers::pi * random.uniform();
        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
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
                 StdComplex(0, 1) * std::sin(angle / 2) * matrix;

        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate pauli_gate = gate::PauliRotation(pauli, angle);
        pauli_gate->update_quantum_state(states);

        test_state = matrix * test_state;

        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (std::uint64_t i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }

        Gate pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(states);
        states_cp = states.get_amplitudes();
        auto states_bef_cp = states_bef.get_amplitudes();

        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (std::uint64_t i = 0; i < dim; i++) {
                ASSERT_NEAR(
                    std::abs((StdComplex)(states_cp[batch_id][i] - states_bef_cp[batch_id][i])),
                    0,
                    eps);
            }
        }
    }
}

void run_random_batched_gate_apply_none_dense(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;
    Eigen::Matrix<StdComplex, 1, 1, Eigen::RowMajor> U;
    Random random;
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
        StateVectorBatched states =
            StateVectorBatched::Haar_random_states(n_qubits, BATCH_SIZE, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        std::vector<std::uint64_t> target_list = {};
        auto re = random.uniform();
        auto im = random.uniform();
        auto val = StdComplex(re, im);
        auto norm = std::sqrt(std::norm(val));
        U(0, 0) = val / norm;
        internal::ComplexMatrix mat = internal::ComplexMatrix::Identity(dim, dim);
        mat *= val / norm;
        std::vector<std::uint64_t> control_list = {};
        Gate dense_gate = gate::DenseMatrix(target_list, U, control_list);
        dense_gate->update_quantum_state(states);
        test_state = mat * test_state;
        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (std::uint64_t i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_batched_gate_apply_single_dense(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;

    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U;
    std::uint64_t target;
    Kokkos::View<Complex**> mat_view("mat_view", 2, 2);
    Random random;
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
        StateVectorBatched states =
            StateVectorBatched::Haar_random_states(n_qubits, BATCH_SIZE, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        target = random.int64() % n_qubits;
        std::vector<std::uint64_t> target_list = {target};
        U = get_eigen_matrix_random_one_target_unitary();
        internal::ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                mat(i, j) = U(i, j);
            }
        }
        std::vector<std::uint64_t> control_list = {};
        Gate dense_gate = gate::DenseMatrix(target_list, mat, control_list);
        dense_gate->update_quantum_state(states);
        test_state = get_expanded_eigen_matrix_with_identity(target, U, n_qubits) * test_state;
        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (std::uint64_t i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_batched_gate_apply_sparse(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    internal::SparseComplexMatrix mat;
    std::vector<std::uint64_t> targets(3);
    std::vector<std::uint64_t> index_list;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> u1, u2, u3;
    Eigen::Matrix<StdComplex, 8, 8, Eigen::RowMajor> Umerge;
    for (std::uint64_t i = 0; i < n_qubits; i++) {
        index_list.push_back(i);
    }
    for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
        StateVectorBatched states =
            StateVectorBatched::Haar_random_states(n_qubits, BATCH_SIZE, true);
        auto state_cp = states.get_state_vector_at(0).get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        std::shuffle(index_list.begin(), index_list.end(), engine);
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        targets[2] = index_list[2];
        u1 = get_eigen_matrix_random_one_target_unitary();
        u2 = get_eigen_matrix_random_one_target_unitary();
        u3 = get_eigen_matrix_random_one_target_unitary();
        std::vector<std::uint64_t> target_list = {targets[0], targets[1], targets[2]};
        std::vector<std::uint64_t> control_list = {};

        test_state = get_expanded_eigen_matrix_with_identity(target_list[2], u3, n_qubits) *
                     get_expanded_eigen_matrix_with_identity(target_list[1], u2, n_qubits) *
                     get_expanded_eigen_matrix_with_identity(target_list[0], u1, n_qubits) *
                     test_state;

        Umerge = internal::kronecker_product(u3, internal::kronecker_product(u2, u1));
        mat = Umerge.sparseView();
        Gate sparse_gate = gate::SparseMatrix(target_list, mat, control_list);
        sparse_gate->update_quantum_state(states);
        auto states_cp = states.get_amplitudes();
        for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
            for (std::uint64_t i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_batched_gate_apply_general_dense(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U1, U2, U3;
    std::vector<std::uint64_t> targets(3);
    std::vector<std::uint64_t> index_list;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (std::uint64_t i = 0; i < n_qubits; i++) {
        index_list.push_back(i);
    }
    // general single
    {
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVectorBatched states =
                StateVectorBatched::Haar_random_states(n_qubits, BATCH_SIZE, true);
            auto state_cp = states.get_state_vector_at(0).get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary();
            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            test_state =
                get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) * test_state;
            std::vector<std::uint64_t> target_list = {targets[0]};
            std::vector<std::uint64_t> control_list = {};
            Gate dense_gate = gate::DenseMatrix(target_list, U1, control_list);
            dense_gate->update_quantum_state(states);
            auto states_cp = states.get_amplitudes();
            for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
                for (std::uint64_t i = 0; i < dim; i++) {
                    ASSERT_NEAR(
                        std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
                }
            }
        }
    }
    // general double
    {
        Eigen::Matrix<StdComplex, 4, 4, Eigen::RowMajor> Umerge;
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVectorBatched states =
                StateVectorBatched::Haar_random_states(n_qubits, BATCH_SIZE, true);
            auto state_cp = states.get_state_vector_at(0).get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary();
            U2 = get_eigen_matrix_random_one_target_unitary();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            Umerge = internal::kronecker_product(U2, U1);
            test_state = get_expanded_eigen_matrix_with_identity(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) *
                         test_state;
            std::vector<std::uint64_t> target_list = {targets[0], targets[1]};
            std::vector<std::uint64_t> control_list = {};
            Gate dense_gate = gate::DenseMatrix(target_list, Umerge, control_list);
            dense_gate->update_quantum_state(states);
            auto states_cp = states.get_amplitudes();
            for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
                for (std::uint64_t i = 0; i < dim; i++) {
                    ASSERT_NEAR(
                        std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
                }
            }
        }
    }
    // general triple
    {
        Eigen::Matrix<StdComplex, 8, 8, Eigen::RowMajor> Umerge;
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVectorBatched states =
                StateVectorBatched::Haar_random_states(n_qubits, BATCH_SIZE, true);
            auto state_cp = states.get_state_vector_at(0).get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary();
            U2 = get_eigen_matrix_random_one_target_unitary();
            U3 = get_eigen_matrix_random_one_target_unitary();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            targets[2] = index_list[2];
            Umerge = internal::kronecker_product(U3, internal::kronecker_product(U2, U1));

            test_state = get_expanded_eigen_matrix_with_identity(targets[2], U3, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) *
                         test_state;
            std::vector<std::uint64_t> target_list = {targets[0], targets[1], targets[2]};
            std::vector<std::uint64_t> control_list = {};
            Gate dense_gate = gate::DenseMatrix(target_list, Umerge, control_list);
            dense_gate->update_quantum_state(states);
            auto states_cp = states.get_amplitudes();
            for (std::uint64_t batch_id = 0; batch_id < states.batch_size(); batch_id++) {
                for (std::uint64_t i = 0; i < dim; i++) {
                    ASSERT_NEAR(
                        std::abs((StdComplex)states_cp[batch_id][i] - test_state[i]), 0, eps);
                }
            }
        }
    }
}

TEST(BatchedGateTest, ApplyI) { run_random_batched_gate_apply<gate::I>(5); }
TEST(BatchedGateTest, ApplyGlobalPhase) { run_random_batched_gate_apply<gate::GlobalPhase>(5); }
TEST(BatchedGateTest, ApplyX) { run_random_batched_gate_apply<gate::X>(5, make_X); }
TEST(BatchedGateTest, ApplyY) { run_random_batched_gate_apply<gate::Y>(5, make_Y); }
TEST(BatchedGateTest, ApplyZ) { run_random_batched_gate_apply<gate::Z>(5, make_Z); }
TEST(BatchedGateTest, ApplyH) { run_random_batched_gate_apply<gate::H>(5, make_H); }
TEST(BatchedGateTest, ApplyS) { run_random_batched_gate_apply<gate::S>(5, make_S); }
TEST(BatchedGateTest, ApplySdag) { run_random_batched_gate_apply<gate::Sdag>(5, make_Sdag); }
TEST(BatchedGateTest, ApplyT) { run_random_batched_gate_apply<gate::T>(5, make_T); }
TEST(BatchedGateTest, ApplyTdag) { run_random_batched_gate_apply<gate::Tdag>(5, make_Tdag); }
TEST(BatchedGateTest, ApplySqrtX) { run_random_batched_gate_apply<gate::SqrtX>(5, make_SqrtX); }
TEST(BatchedGateTest, ApplySqrtY) { run_random_batched_gate_apply<gate::SqrtY>(5, make_SqrtY); }
TEST(BatchedGateTest, ApplySqrtXdag) {
    run_random_batched_gate_apply<gate::SqrtXdag>(5, make_SqrtXdag);
}
TEST(BatchedGateTest, ApplySqrtYdag) {
    run_random_batched_gate_apply<gate::SqrtYdag>(5, make_SqrtYdag);
}
TEST(BatchedGateTest, ApplyP0) { run_random_batched_gate_apply<gate::P0>(5, make_P0); }
TEST(BatchedGateTest, ApplyP1) { run_random_batched_gate_apply<gate::P1>(5, make_P1); }
TEST(BatchedGateTest, ApplyRX) { run_random_batched_gate_apply<gate::RX>(5, make_RX); }
TEST(BatchedGateTest, ApplyRY) { run_random_batched_gate_apply<gate::RY>(5, make_RY); }
TEST(BatchedGateTest, ApplyRZ) { run_random_batched_gate_apply<gate::RZ>(5, make_RZ); }
TEST(BatchedGateTest, ApplyIBMQ) { run_random_batched_gate_apply_IBMQ(5, make_U); }
TEST(BatchedGateTest, ApplySparseMatrixGate) { run_random_batched_gate_apply_sparse(6); }
TEST(BatchedGateTest, ApplyDenseMatrixGate) {
    run_random_batched_gate_apply_none_dense(6);
    run_random_batched_gate_apply_single_dense(6);
    run_random_batched_gate_apply_general_dense(6);
}
TEST(BatchedGateTest, ApplyPauliGate) { run_random_batched_gate_apply_pauli(5); }

TEST(BatchedGateTest, ApplyProbablisticGate) {
    auto probgate = gate::Probablistic({.1, .9}, {gate::X(0), gate::I()});
    StateVectorBatched states(BATCH_SIZE, 1);
    std::vector<std::vector<std::uint64_t>> befores, afters;
    std::vector<std::uint64_t> x_counts(BATCH_SIZE), i_counts(BATCH_SIZE);
    for ([[maybe_unused]] auto _ : std::views::iota(0, 100)) {
        befores = states.sampling(1);
        probgate->update_quantum_state(states);
        afters = states.sampling(1);
        for (std::size_t i = 0; i < BATCH_SIZE; i++) {
            if (befores[i][0] != afters[i][0]) {
                x_counts[i]++;
            } else {
                i_counts[i]++;
            }
        }
    }
    // These test is probablistic, but pass at least 99.9% cases.
    for (std::size_t i = 0; i < BATCH_SIZE; i++) {
        ASSERT_GT(x_counts[i], 0);
        ASSERT_GT(i_counts[i], 0);
        ASSERT_LT(x_counts[i], i_counts[i]);
    }
}

void test_batched_gate(Gate gate_control,
                       Gate gate_simple,
                       std::uint64_t n_qubits,
                       std::uint64_t control_mask) {
    StateVectorBatched states = StateVectorBatched::Haar_random_states(BATCH_SIZE, n_qubits, true);
    auto amplitudes = states.get_amplitudes();
    StateVectorBatched state_controlled(BATCH_SIZE, n_qubits - std::popcount(control_mask));
    std::vector<std::vector<Complex>> amplitudes_controlled(
        BATCH_SIZE, std::vector<Complex>(state_controlled.dim()));
    for (std::size_t i = 0; i < BATCH_SIZE; i++) {
        for (std::uint64_t j = 0; j < state_controlled.dim(); j++) {
            amplitudes_controlled[i][j] =
                amplitudes[i]
                          [internal::insert_zero_at_mask_positions(j, control_mask) | control_mask];
        }
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(states);
    gate_simple->update_quantum_state(state_controlled);
    amplitudes = states.get_amplitudes();
    amplitudes_controlled = state_controlled.get_amplitudes();
    for (std::size_t i = 0; i < BATCH_SIZE; i++) {
        for (std::uint64_t j = 0; j < state_controlled.dim(); j++) {
            ASSERT_NEAR(
                Kokkos::abs(amplitudes_controlled[i][j] -
                            amplitudes[i][internal::insert_zero_at_mask_positions(j, control_mask) |
                                          control_mask]),
                0.,
                eps);
        }
    }
}

template <std::uint64_t num_target, std::uint64_t num_rotation, typename Factory>
void test_batched_standard_gate_control(Factory factory, std::uint64_t n) {
    Random random;
    std::vector<std::uint64_t> shuffled(n);
    std::iota(shuffled.begin(), shuffled.end(), 0ULL);
    for (std::uint64_t i : std::views::iota(0ULL, n) | std::views::reverse) {
        std::uint64_t j = random.int32() % (i + 1);
        if (i != j) std::swap(shuffled[i], shuffled[j]);
    }
    std::vector<std::uint64_t> targets(num_target);
    for (std::uint64_t i : std::views::iota(0ULL, num_target)) {
        targets[i] = shuffled[i];
    }
    std::uint64_t num_control = random.int32() % (n - num_target + 1);
    std::vector<std::uint64_t> controls(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[num_target + i];
    }
    std::uint64_t control_mask = 0ULL;
    for (std::uint64_t c : controls) control_mask |= 1ULL << c;
    std::vector<double> angles(num_rotation);
    for (double& angle : angles) angle = random.uniform() * std::numbers::pi * 2;
    if constexpr (num_target == 0 && num_rotation == 1) {
        Gate g1 = factory(angles[0], controls);
        Gate g2 = factory(angles[0], {});
        test_batched_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 0) {
        Gate g1 = factory(targets[0], controls);
        Gate g2 =
            factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)), {});
        test_batched_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 1) {
        Gate g1 = factory(targets[0], angles[0], controls);
        Gate g2 = factory(
            targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)), angles[0], {});
        test_batched_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 2) {
        Gate g1 = factory(targets[0], angles[0], angles[1], controls);
        Gate g2 = factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                          angles[0],
                          angles[1],
                          {});
        test_batched_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 3) {
        Gate g1 = factory(targets[0], angles[0], angles[1], angles[2], controls);
        Gate g2 = factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                          angles[0],
                          angles[1],
                          angles[2],
                          {});
        test_batched_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 2 && num_rotation == 0) {
        Gate g1 = factory(targets[0], targets[1], controls);
        Gate g2 = factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                          targets[1] - std::popcount(control_mask & ((1ULL << targets[1]) - 1)),
                          {});
        test_batched_gate(g1, g2, n, control_mask);
    } else {
        FAIL();
    }
}

template <bool rotation>
void test_batched_pauli_control(std::uint64_t n) {
    PauliOperator::Data data1, data2;
    std::vector<std::uint64_t> controls;
    std::uint64_t control_mask = 0;
    std::uint64_t num_control = 0;
    Random random;
    for (std::uint64_t i : std::views::iota(0ULL, n)) {
        std::uint64_t dat = random.int32() % 12;
        if (dat < 4) {
            data1.add_single_pauli(i, dat);
            data2.add_single_pauli(i - num_control, dat);
        } else if (dat < 8) {
            controls.push_back(i);
            control_mask |= 1ULL << i;
            num_control++;
        }
    }
    if constexpr (!rotation) {
        Gate g1 = gate::Pauli(PauliOperator(data1), controls);
        Gate g2 = gate::Pauli(PauliOperator(data2), {});
        test_batched_gate(g1, g2, n, control_mask);
    } else {
        double angle = random.uniform() * std::numbers::pi * 2;
        Gate g1 = gate::PauliRotation(PauliOperator(data1), angle, controls);
        Gate g2 = gate::PauliRotation(PauliOperator(data2), angle, {});
        test_batched_gate(g1, g2, n, control_mask);
    }
}

template <std::uint64_t num_target>
void test_batched_matrix_control(std::uint64_t n_qubits) {
    Random random;
    std::vector<std::uint64_t> shuffled(n_qubits);
    std::iota(shuffled.begin(), shuffled.end(), 0ULL);
    for (std::uint64_t i : std::views::iota(0ULL, n_qubits) | std::views::reverse) {
        std::uint64_t j = random.int32() % (i + 1);
        if (i != j) std::swap(shuffled[i], shuffled[j]);
    }
    std::vector<std::uint64_t> targets(num_target);
    for (std::uint64_t i : std::views::iota(0ULL, num_target)) {
        targets[i] = shuffled[i];
    }
    std::uint64_t num_control = random.int32() % (n_qubits - num_target + 1);
    std::vector<std::uint64_t> controls(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[num_target + i];
    }
    std::uint64_t control_mask = 0ULL;
    for (std::uint64_t c : controls) control_mask |= 1ULL << c;
    auto adjust = [](std::vector<std::uint64_t> targets, std::uint64_t control_mask) {
        std::vector<std::uint64_t> new_targets;
        for (auto i : targets) {
            new_targets.push_back(i - std::popcount(control_mask & ((1ULL << i) - 1)));
        }
        return new_targets;
    };
    auto new_targets = adjust(targets, control_mask);
    if constexpr (num_target == 0) {
        auto re = random.uniform();
        auto im = random.uniform();
        auto val = StdComplex(re, im);
        auto norm = std::sqrt(std::norm(val));
        internal::ComplexMatrix mat(1, 1);
        mat(0, 0) = val / norm;
        Gate d1 = gate::DenseMatrix(targets, mat, controls);
        Gate d2 = gate::DenseMatrix(new_targets, mat, {});
        Gate s1 = gate::SparseMatrix(targets, mat.sparseView(), controls);
        Gate s2 = gate::SparseMatrix(new_targets, mat.sparseView(), {});
        test_batched_gate(d1, d2, n_qubits, control_mask);
        test_batched_gate(s1, s2, n_qubits, control_mask);
    } else if constexpr (num_target == 1) {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U =
            get_eigen_matrix_random_one_target_unitary();
        internal::ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate d1 = gate::DenseMatrix(targets, mat, controls);
        Gate d2 = gate::DenseMatrix(new_targets, mat, {});
        Gate s1 = gate::SparseMatrix(targets, mat.sparseView(), controls);
        Gate s2 = gate::SparseMatrix(new_targets, mat.sparseView(), {});
        test_batched_gate(d1, d2, n_qubits, control_mask);
        test_batched_gate(s1, s2, n_qubits, control_mask);
    } else if constexpr (num_target == 2) {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U1 =
            get_eigen_matrix_random_one_target_unitary();
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U2 =
            get_eigen_matrix_random_one_target_unitary();
        auto U = internal::kronecker_product(U2, U1);
        internal::ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate d1 = gate::DenseMatrix(targets, mat, controls);
        Gate d2 = gate::DenseMatrix(new_targets, mat, {});
        Gate s1 = gate::SparseMatrix(targets, mat.sparseView(), controls);
        Gate s2 = gate::SparseMatrix(new_targets, mat.sparseView(), {});
        test_batched_gate(d1, d2, n_qubits, control_mask);
        test_batched_gate(s1, s2, n_qubits, control_mask);
    } else {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U1 =
            get_eigen_matrix_random_one_target_unitary();
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U2 =
            get_eigen_matrix_random_one_target_unitary();
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U3 =
            get_eigen_matrix_random_one_target_unitary();
        auto U = internal::kronecker_product(U3, internal::kronecker_product(U2, U1));
        internal::ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate d1 = gate::DenseMatrix(targets, mat, controls);
        Gate d2 = gate::DenseMatrix(new_targets, mat, {});
        Gate s1 = gate::SparseMatrix(targets, mat.sparseView(), controls);
        Gate s2 = gate::SparseMatrix(new_targets, mat.sparseView(), {});
        test_batched_gate(d1, d2, n_qubits, control_mask);
        test_batched_gate(s1, s2, n_qubits, control_mask);
    }
}

TEST(BatchGateTest, Control) {
    std::uint64_t n = 10;
    for ([[maybe_unused]] std::uint64_t _ : std::views::iota(0, 10)) {
        test_batched_standard_gate_control<0, 1>(gate::GlobalPhase, n);
        test_batched_standard_gate_control<1, 0>(gate::X, n);
        test_batched_standard_gate_control<1, 0>(gate::Y, n);
        test_batched_standard_gate_control<1, 0>(gate::Z, n);
        test_batched_standard_gate_control<1, 0>(gate::S, n);
        test_batched_standard_gate_control<1, 0>(gate::Sdag, n);
        test_batched_standard_gate_control<1, 0>(gate::T, n);
        test_batched_standard_gate_control<1, 0>(gate::Tdag, n);
        test_batched_standard_gate_control<1, 0>(gate::SqrtX, n);
        test_batched_standard_gate_control<1, 0>(gate::SqrtXdag, n);
        test_batched_standard_gate_control<1, 0>(gate::SqrtY, n);
        test_batched_standard_gate_control<1, 0>(gate::SqrtYdag, n);
        test_batched_standard_gate_control<1, 0>(gate::P0, n);
        test_batched_standard_gate_control<1, 0>(gate::P1, n);
        test_batched_standard_gate_control<1, 1>(gate::RX, n);
        test_batched_standard_gate_control<1, 1>(gate::RY, n);
        test_batched_standard_gate_control<1, 1>(gate::RZ, n);
        test_batched_standard_gate_control<1, 1>(gate::U1, n);
        test_batched_standard_gate_control<1, 2>(gate::U2, n);
        test_batched_standard_gate_control<1, 3>(gate::U3, n);
        test_batched_standard_gate_control<2, 0>(gate::Swap, n);
        test_batched_pauli_control<false>(n);
        test_batched_pauli_control<true>(n);
        test_batched_matrix_control<0>(n);
        test_batched_matrix_control<1>(n);
        test_batched_matrix_control<2>(n);
        test_batched_matrix_control<3>(n);
    }
}
