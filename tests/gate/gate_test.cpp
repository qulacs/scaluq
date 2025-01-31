#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <scaluq/gate/gate_factory.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

#define FLOAT_AND_SPACE(Fp, Sp) template <std::floating_point Fp, ExecutionSpace Sp>

template <std::floating_point Fp, ExecutionSpace Sp, Gate<Fp, Sp> (*QuantumGateConstructor)()>
void run_random_gate_apply(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const Gate<Fp, Sp> gate = QuantumGateConstructor();
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state = test_state;

        for (int i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <std::floating_point Fp,
          ExecutionSpace Sp,
          Gate<Fp, Sp> (*QuantumGateConstructor)(Fp, const std::vector<std::uint64_t>&)>
void run_random_gate_apply(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const Fp angle = std::numbers::pi * random.uniform();
        const Gate<Fp, Sp> gate = QuantumGateConstructor(angle, {});
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state = std::polar<Fp>(1, angle) * test_state;

        for (int i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <std::floating_point Fp,
          ExecutionSpace Sp,
          Gate<Fp, Sp> (*QuantumGateConstructor)(std::uint64_t, const std::vector<std::uint64_t>&)>
void run_random_gate_apply(std::uint64_t n_qubits,
                           std::function<ComplexMatrix<Fp>()> matrix_factory) {
    const auto matrix = matrix_factory();
    const int dim = 1ULL << n_qubits;
    Random random;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const std::uint64_t target = random.int64() % n_qubits;
        const Gate<Fp, Sp> gate = QuantumGateConstructor(target, {});
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state =
            get_expanded_eigen_matrix_with_identity<Fp>(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <
    std::floating_point Fp,
    ExecutionSpace Sp,
    Gate<Fp, Sp> (*QuantumGateConstructor)(std::uint64_t, Fp, const std::vector<std::uint64_t>&)>
void run_random_gate_apply(std::uint64_t n_qubits,
                           std::function<ComplexMatrix<Fp>(Fp)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const Fp angle = std::numbers::pi * random.uniform();
        const auto matrix = matrix_factory(angle);
        const std::uint64_t target = random.int64() % n_qubits;
        const Gate<Fp, Sp> gate = QuantumGateConstructor(target, angle, {});
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state =
            get_expanded_eigen_matrix_with_identity<Fp>(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <std::floating_point Fp, ExecutionSpace Sp>
void run_random_gate_apply_IBMQ(std::uint64_t n_qubits,
                                std::function<ComplexMatrix<Fp>(Fp, Fp, Fp)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int gate_type = 0; gate_type < 3; gate_type++) {
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            Fp theta = std::numbers::pi * random.uniform();
            Fp phi = std::numbers::pi * random.uniform();
            Fp lambda = std::numbers::pi * random.uniform();
            if (gate_type == 0) {
                theta = 0;
                phi = 0;
            } else if (gate_type == 1) {
                theta = std::numbers::pi / 2;
            }
            const auto matrix = matrix_factory(theta, phi, lambda);
            const std::uint64_t target = random.int64() % n_qubits;
            Gate<Fp, Sp> gate;
            if (gate_type == 0) {
                gate = gate::U1<Fp, Sp>(target, lambda, {});
            } else if (gate_type == 1) {
                gate = gate::U2<Fp, Sp>(target, phi, lambda, {});
            } else {
                gate = gate::U3<Fp, Sp>(target, theta, phi, lambda, {});
            }
            gate->update_quantum_state(state);
            state_cp = state.get_amplitudes();

            test_state =
                get_expanded_eigen_matrix_with_identity<Fp>(target, matrix, n_qubits) * test_state;

            for (int i = 0; i < dim; i++) {
                check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
            }
        }
    }
}

template <std::floating_point Fp, ExecutionSpace Sp>
void run_random_gate_apply_two_target(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    std::function<ComplexMatrix<Fp>(std::uint64_t, std::uint64_t, std::uint64_t)> func_eig;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        for (int g = 0; g < 2; g++) {
            Gate<Fp, Sp> gate;
            auto state_cp = state.get_amplitudes();
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            std::uint64_t target = random.int64() % n_qubits;
            std::uint64_t control = random.int64() % n_qubits;
            if (target == control) target = (target + 1) % n_qubits;
            if (g == 0) {
                gate = gate::CX<Fp>(control, target);
                func_eig = get_eigen_matrix_full_qubit_CX<Fp>;
            } else {
                gate = gate::CZ<Fp>(control, target);
                func_eig = get_eigen_matrix_full_qubit_CZ<Fp>;
            }
            gate->update_quantum_state(state);
            state_cp = state.get_amplitudes();

            ComplexMatrix<Fp> test_mat = func_eig(control, target, n_qubits);
            test_state = test_mat * test_state;

            for (int i = 0; i < dim; i++) {
                check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
            }
        }
    }

    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::uint64_t target1 = random.int64() % n_qubits;
        std::uint64_t target2 = random.int64() % n_qubits;
        if (target1 == target2) target1 = (target1 + 1) % n_qubits;
        auto gate = gate::Swap<Fp>(target1, target2);
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        ComplexMatrix<Fp> test_mat =
            get_eigen_matrix_full_qubit_Swap<Fp>(target1, target2, n_qubits);
        test_state = test_mat * test_state;

        for (int i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <std::floating_point Fp, ExecutionSpace Sp>
void run_random_gate_apply_none_dense(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;
    Eigen::Matrix<StdComplex<Fp>, 1, 1, Eigen::RowMajor> U;
    Random random;
    for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
        StateVector state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        std::vector<std::uint64_t> target_list = {};
        auto re = random.uniform();
        auto im = random.uniform();
        auto val = StdComplex<Fp>(re, im);
        Fp norm = std::sqrt(std::norm(val));
        U(0, 0) = val / norm;
        // matは全ての要素がval/normの対角行列
        ComplexMatrix<Fp> mat = ComplexMatrix<Fp>::Identity(dim, dim);
        mat *= val / norm;
        std::vector<std::uint64_t> control_list = {};
        Gate<Fp, Sp> dense_gate = gate::DenseMatrix<Fp, Sp>(target_list, U, control_list);
        dense_gate->update_quantum_state(state);
        test_state = mat * test_state;
        state_cp = state.get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <std::floating_point Fp, ExecutionSpace Sp>
void run_random_gate_apply_single_dense(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;

    Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U;
    std::uint64_t target;
    Kokkos::View<Complex<Fp>**> mat_view("mat_view", 2, 2);
    Random random;
    for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
        StateVector state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        target = random.int64() % n_qubits;
        std::vector<std::uint64_t> target_list = {target};
        U = get_eigen_matrix_random_one_target_unitary<Fp>();
        ComplexMatrix<Fp> mat(U.rows(), U.cols());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                mat(i, j) = U(i, j);
            }
        }
        std::vector<std::uint64_t> control_list = {};
        Gate<Fp, Sp> dense_gate = gate::DenseMatrix<Fp, Sp>(target_list, mat, control_list);
        dense_gate->update_quantum_state(state);
        test_state = get_expanded_eigen_matrix_with_identity<Fp>(target, U, n_qubits) * test_state;
        state_cp = state.get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <std::floating_point Fp, ExecutionSpace Sp>
void run_random_gate_apply_general_dense(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U1, U2, U3;
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
            StateVector<Fp, Sp> state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
            auto state_cp = state.get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary<Fp>();
            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            test_state =
                get_expanded_eigen_matrix_with_identity<Fp>(targets[0], U1, n_qubits) * test_state;
            std::vector<std::uint64_t> target_list = {targets[0]};
            std::vector<std::uint64_t> control_list = {};
            Gate<Fp, Sp> dense_gate = gate::DenseMatrix<Fp, Sp>(target_list, U1, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
            }
        }
    }
    // general double
    {
        Eigen::Matrix<StdComplex<Fp>, 4, 4, Eigen::RowMajor> Umerge;
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVector<Fp, Sp> state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
            auto state_cp = state.get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary<Fp>();
            U2 = get_eigen_matrix_random_one_target_unitary<Fp>();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            Umerge = internal::kronecker_product<Fp>(U2, U1);
            test_state = get_expanded_eigen_matrix_with_identity<Fp>(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity<Fp>(targets[0], U1, n_qubits) *
                         test_state;
            std::vector<std::uint64_t> target_list = {targets[0], targets[1]};
            std::vector<std::uint64_t> control_list = {};
            Gate<Fp, Sp> dense_gate = gate::DenseMatrix<Fp, Sp>(target_list, Umerge, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
            }
        }
    }
    // general triple
    {
        Eigen::Matrix<StdComplex<Fp>, 8, 8, Eigen::RowMajor> Umerge;
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVector<Fp, Sp> state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
            auto state_cp = state.get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary<Fp>();
            U2 = get_eigen_matrix_random_one_target_unitary<Fp>();
            U3 = get_eigen_matrix_random_one_target_unitary<Fp>();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            targets[2] = index_list[2];
            Umerge = internal::kronecker_product<Fp>(U3, internal::kronecker_product<Fp>(U2, U1));

            test_state = get_expanded_eigen_matrix_with_identity<Fp>(targets[2], U3, n_qubits) *
                         get_expanded_eigen_matrix_with_identity<Fp>(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity<Fp>(targets[0], U1, n_qubits) *
                         test_state;
            std::vector<std::uint64_t> target_list = {targets[0], targets[1], targets[2]};
            std::vector<std::uint64_t> control_list = {};
            Gate<Fp, Sp> dense_gate = gate::DenseMatrix<Fp, Sp>(target_list, Umerge, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.get_amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
            }
        }
    }
}

template <std::floating_point Fp, ExecutionSpace Sp>
void run_random_gate_apply_sparse(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;

    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    internal::SparseComplexMatrix<Fp> mat;
    std::vector<std::uint64_t> targets(3);
    std::vector<std::uint64_t> index_list;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> u1, u2, u3;
    Eigen::Matrix<StdComplex<Fp>, 8, 8, Eigen::RowMajor> Umerge;
    for (std::uint64_t i = 0; i < n_qubits; i++) {
        index_list.push_back(i);
    }
    for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
        StateVector<Fp, Sp> state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        std::shuffle(index_list.begin(), index_list.end(), engine);
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        targets[2] = index_list[2];
        u1 = get_eigen_matrix_random_one_target_unitary<Fp>();
        u2 = get_eigen_matrix_random_one_target_unitary<Fp>();
        u3 = get_eigen_matrix_random_one_target_unitary<Fp>();
        std::vector<std::uint64_t> target_list = {targets[0], targets[1], targets[2]};
        std::vector<std::uint64_t> control_list = {};

        test_state = get_expanded_eigen_matrix_with_identity<Fp>(target_list[2], u3, n_qubits) *
                     get_expanded_eigen_matrix_with_identity<Fp>(target_list[1], u2, n_qubits) *
                     get_expanded_eigen_matrix_with_identity<Fp>(target_list[0], u1, n_qubits) *
                     test_state;

        Umerge = internal::kronecker_product<Fp>(u3, internal::kronecker_product<Fp>(u2, u1));
        mat = Umerge.sparseView();
        Gate<Fp, Sp> sparse_gate = gate::SparseMatrix<Fp, Sp>(target_list, mat, control_list);
        sparse_gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
    }
}

template <std::floating_point Fp, ExecutionSpace Sp>
void run_random_gate_apply_pauli(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;
    ComplexVector<Fp> test_state = ComplexVector<Fp>::Zero(dim);
    ComplexMatrix<Fp> matrix;

    // Test for PauliGate
    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector<Fp, Sp> state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        auto state_bef = state.copy();

        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

        if (pauli_id_vec[0] == 0) {
            matrix = make_I<Fp>();
        } else if (pauli_id_vec[0] == 1) {
            matrix = make_X<Fp>();
        } else if (pauli_id_vec[0] == 2) {
            matrix = make_Y<Fp>();
        } else if (pauli_id_vec[0] == 3) {
            matrix = make_Z<Fp>();
        }
        for (int i = 1; i < (int)n_qubits; i++) {
            if (pauli_id_vec[i] == 0) {
                matrix = internal::kronecker_product<Fp>(make_I<Fp>(), matrix);
            } else if (pauli_id_vec[i] == 1) {
                matrix = internal::kronecker_product<Fp>(make_X<Fp>(), matrix);
            } else if (pauli_id_vec[i] == 2) {
                matrix = internal::kronecker_product<Fp>(make_Y<Fp>(), matrix);
            } else if (pauli_id_vec[i] == 3) {
                matrix = internal::kronecker_product<Fp>(make_Z<Fp>(), matrix);
            }
        }

        PauliOperator<Fp, Sp> pauli(target_vec, pauli_id_vec, 1.0);
        auto pauli_gate = gate::Pauli<Fp>(pauli);
        pauli_gate->update_quantum_state(state);

        state_cp = state.get_amplitudes();
        test_state = matrix * test_state;

        // check if the state is updated correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }

        auto state_bef_cp = state_bef.get_amplitudes();
        auto pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], (StdComplex<Fp>)state_bef_cp[i]);
        }
    }

    // Test for PauliRotationGate
    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector<Fp, Sp> state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        auto state_bef = state.copy();
        assert(test_state.size() == (int)state_cp.size());
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        const Fp angle = std::numbers::pi * random.uniform();
        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

        if (pauli_id_vec[0] == 0) {
            matrix = make_I<Fp>();
        } else if (pauli_id_vec[0] == 1) {
            matrix = make_X<Fp>();
        } else if (pauli_id_vec[0] == 2) {
            matrix = make_Y<Fp>();
        } else if (pauli_id_vec[0] == 3) {
            matrix = make_Z<Fp>();
        }
        for (int i = 1; i < (int)n_qubits; i++) {
            if (pauli_id_vec[i] == 0) {
                matrix = internal::kronecker_product<Fp>(make_I<Fp>(), matrix);
            } else if (pauli_id_vec[i] == 1) {
                matrix = internal::kronecker_product<Fp>(make_X<Fp>(), matrix);
            } else if (pauli_id_vec[i] == 2) {
                matrix = internal::kronecker_product<Fp>(make_Y<Fp>(), matrix);
            } else if (pauli_id_vec[i] == 3) {
                matrix = internal::kronecker_product<Fp>(make_Z<Fp>(), matrix);
            }
        }
        matrix = std::cos(angle / 2) * ComplexMatrix<Fp>::Identity(dim, dim) -
                 StdComplex<Fp>(0, 1) * std::sin(angle / 2) * matrix;
        PauliOperator<Fp, Sp> pauli(target_vec, pauli_id_vec, 1.0);
        Gate<Fp, Sp> pauli_gate = gate::PauliRotation(pauli, angle);
        pauli_gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();
        test_state = matrix * test_state;
        assert((int)state_cp.size() == test_state.size());
        // check if the state is updated correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], test_state[i]);
        }
        auto pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(state);
        state_cp = state.get_amplitudes();
        auto state_bef_cp = state_bef.get_amplitudes();
        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near((StdComplex<Fp>)state_cp[i], (StdComplex<Fp>)state_bef_cp[i]);
        }
    }
}

#define EXECUTE_GATE_TEST(GateName, arg)                                  \
    TEST(GateTest, Apply##GateName) {                                     \
        run_random_gate_apply<double, DefaultSpace, gate::GateName>(arg); \
        run_random_gate_apply<float, DefaultSpace, gate::GateName>(arg);  \
        run_random_gate_apply<double, HostSpace, gate::GateName>(arg);    \
        run_random_gate_apply<float, HostSpace, gate::GateName>(arg);     \
    }
EXECUTE_GATE_TEST(I, 5)
EXECUTE_GATE_TEST(GlobalPhase, 5)
#undef EXECUTE_GATE_TEST

#define EXECUTE_GATE_TEST(GateName, arg)                                                           \
    TEST(GateTest, Apply##GateName) {                                                              \
        run_random_gate_apply<double, DefaultSpace, gate::GateName>(arg, make_##GateName<double>); \
        run_random_gate_apply<float, DefaultSpace, gate::GateName>(arg, make_##GateName<float>);   \
        run_random_gate_apply<double, HostSpace, gate::GateName>(arg, make_##GateName<double>);    \
        run_random_gate_apply<float, HostSpace, gate::GateName>(arg, make_##GateName<float>);      \
    }
EXECUTE_GATE_TEST(X, 5)
EXECUTE_GATE_TEST(Y, 5)
EXECUTE_GATE_TEST(Z, 5)
EXECUTE_GATE_TEST(H, 5)
EXECUTE_GATE_TEST(S, 5)
EXECUTE_GATE_TEST(Sdag, 5)
EXECUTE_GATE_TEST(T, 5)
EXECUTE_GATE_TEST(Tdag, 5)
EXECUTE_GATE_TEST(SqrtX, 5)
EXECUTE_GATE_TEST(SqrtY, 5)
EXECUTE_GATE_TEST(SqrtXdag, 5)
EXECUTE_GATE_TEST(SqrtYdag, 5)
EXECUTE_GATE_TEST(P0, 5)
EXECUTE_GATE_TEST(P1, 5)
EXECUTE_GATE_TEST(RX, 5)
EXECUTE_GATE_TEST(RY, 5)
EXECUTE_GATE_TEST(RZ, 5)
TEST(GateTest, ApplyIBMQ) {
    run_random_gate_apply_IBMQ<double, DefaultSpace>(5, make_U<double>);
    run_random_gate_apply_IBMQ<double, HostSpace>(5, make_U<double>);
    run_random_gate_apply_IBMQ<float, DefaultSpace>(5, make_U<float>);
    run_random_gate_apply_IBMQ<float, HostSpace>(5, make_U<float>);
}
TEST(GateTest, ApplySparseMatrixGate) {
    run_random_gate_apply_sparse<double, DefaultSpace>(6);
    run_random_gate_apply_sparse<double, HostSpace>(6);
    run_random_gate_apply_sparse<float, DefaultSpace>(6);
    run_random_gate_apply_sparse<float, HostSpace>(6);
}
TEST(GateTest, ApplyDenseMatrixGate) {
    run_random_gate_apply_none_dense<double, DefaultSpace>(6);
    run_random_gate_apply_single_dense<double, DefaultSpace>(6);
    run_random_gate_apply_general_dense<double, DefaultSpace>(6);
    run_random_gate_apply_none_dense<double, HostSpace>(6);
    run_random_gate_apply_single_dense<double, HostSpace>(6);
    run_random_gate_apply_general_dense<double, HostSpace>(6);
    run_random_gate_apply_none_dense<float, DefaultSpace>(6);
    run_random_gate_apply_single_dense<float, DefaultSpace>(6);
    run_random_gate_apply_general_dense<float, DefaultSpace>(6);
    run_random_gate_apply_none_dense<float, HostSpace>(6);
    run_random_gate_apply_single_dense<float, HostSpace>(6);
    run_random_gate_apply_general_dense<float, HostSpace>(6);
}

TEST(GateTest, ApplyPauliGate) {
    run_random_gate_apply_pauli<double, DefaultSpace>(6);
    run_random_gate_apply_pauli<float, DefaultSpace>(6);
    run_random_gate_apply_pauli<double, HostSpace>(6);
    run_random_gate_apply_pauli<float, HostSpace>(6);
}

FLOAT_AND_SPACE(Fp, Sp)
void TestProbablisticGate(std::uint64_t n_qubits) {
    std::vector<Fp> distribution = {.1, .9};
    const std::vector<Gate<Fp, Sp>> gate_list = {gate::X<Fp, Sp>(0), gate::I<Fp, Sp>()};
    auto probgate = gate::Probablistic<Fp, Sp>(distribution, gate_list);
    std::uint64_t x_cnt = 0, i_cnt = 0;
    StateVector<Fp, Sp> state(n_qubits);
    for ([[maybe_unused]] auto _ : std::views::iota(0, 100)) {
        std::uint64_t before = state.sampling(1)[0];
        probgate->update_quantum_state(state);
        std::uint64_t after = state.sampling(1)[0];
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
TEST(GateTest, ApplyProbablisticGate) {
    TestProbablisticGate<double, DefaultSpace>(6);
    TestProbablisticGate<double, HostSpace>(6);
    TestProbablisticGate<float, DefaultSpace>(6);
    TestProbablisticGate<float, HostSpace>(6);
}

FLOAT_AND_SPACE(Fp, Sp)
void test_gate(Gate<Fp, Sp> gate_control,
               Gate<Fp, Sp> gate_simple,
               std::uint64_t n_qubits,
               std::uint64_t control_mask) {
    StateVector<Fp, Sp> state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
    auto amplitudes = state.get_amplitudes();
    StateVector<Fp, Sp> state_controlled(n_qubits - std::popcount(control_mask));
    std::vector<Complex<Fp>> amplitudes_controlled(state_controlled.dim());
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        amplitudes_controlled[i] =
            amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) | control_mask];
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(state);
    gate_simple->update_quantum_state(state_controlled);
    amplitudes = state.get_amplitudes();
    amplitudes_controlled = state_controlled.get_amplitudes();
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        check_near(
            (StdComplex<Fp>)amplitudes_controlled[i],
            (StdComplex<Fp>)amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                                       control_mask]);
    }
}

template <std::floating_point Fp,
          ExecutionSpace Sp,
          std::uint64_t num_target,
          std::uint64_t num_rotation,
          typename Factory>
void test_standard_gate_control(Factory factory, std::uint64_t n) {
    Random random;
    std::vector<std::uint64_t> shuffled = random.permutation(n);
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
    std::vector<Fp> angles(num_rotation);
    for (Fp& angle : angles) angle = random.uniform() * std::numbers::pi * 2;
    if constexpr (num_target == 0 && num_rotation == 1) {
        Gate<Fp, Sp> g1 = factory(angles[0], controls);
        Gate<Fp, Sp> g2 = factory(angles[0], {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 0) {
        Gate<Fp, Sp> g1 = factory(targets[0], controls);
        Gate<Fp, Sp> g2 =
            factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)), {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 1) {
        Gate<Fp, Sp> g1 = factory(targets[0], angles[0], controls);
        Gate<Fp, Sp> g2 = factory(
            targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)), angles[0], {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 2) {
        Gate<Fp, Sp> g1 = factory(targets[0], angles[0], angles[1], controls);
        Gate<Fp, Sp> g2 =
            factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                    angles[0],
                    angles[1],
                    {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 3) {
        Gate<Fp, Sp> g1 = factory(targets[0], angles[0], angles[1], angles[2], controls);
        Gate<Fp, Sp> g2 =
            factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                    angles[0],
                    angles[1],
                    angles[2],
                    {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 2 && num_rotation == 0) {
        Gate<Fp, Sp> g1 = factory(targets[0], targets[1], controls);
        Gate<Fp, Sp> g2 =
            factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                    targets[1] - std::popcount(control_mask & ((1ULL << targets[1]) - 1)),
                    {});
        test_gate(g1, g2, n, control_mask);
    } else {
        FAIL();
    }
}

template <std::floating_point Fp, ExecutionSpace Sp, bool rotation>
void test_pauli_control(std::uint64_t n) {
    typename PauliOperator<Fp, Sp>::Data data1, data2;
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
        Gate<Fp, Sp> g1 = gate::Pauli(PauliOperator<Fp, Sp>(data1), controls);
        Gate<Fp, Sp> g2 = gate::Pauli(PauliOperator<Fp, Sp>(data2), {});
        test_gate(g1, g2, n, control_mask);
    } else {
        Fp angle = random.uniform() * std::numbers::pi * 2;
        Gate<Fp, Sp> g1 = gate::PauliRotation(PauliOperator<Fp, Sp>(data1), angle, controls);
        Gate<Fp, Sp> g2 = gate::PauliRotation(PauliOperator<Fp, Sp>(data2), angle, {});
        test_gate(g1, g2, n, control_mask);
    }
}

template <std::floating_point Fp, ExecutionSpace Sp, std::uint64_t num_target>
void test_matrix_control(std::uint64_t n_qubits) {
    Random random;
    std::vector<std::uint64_t> shuffled = random.permutation(n_qubits);
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
        auto val = StdComplex<Fp>(re, im);
        auto norm = std::sqrt(std::norm(val));
        ComplexMatrix<Fp> mat(1, 1);
        mat(0, 0) = val / norm;
        Gate<Fp, Sp> d1 = PauliOperator<Fp, Sp>(targets, mat, controls);
        Gate<Fp, Sp> d2 = PauliOperator<Fp, Sp>(new_targets, mat, {});
        Gate<Fp, Sp> s1 = gate::SparseMatrix<Fp>(targets, mat.sparseView(), controls);
        Gate<Fp, Sp> s2 = gate::SparseMatrix<Fp>(new_targets, mat.sparseView(), {});
        test_gate<Fp>(d1, d2, n_qubits, control_mask);
        test_gate<Fp>(s1, s2, n_qubits, control_mask);
    } else if constexpr (num_target == 1) {
        Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U =
            get_eigen_matrix_random_one_target_unitary<Fp>();
        ComplexMatrix<Fp> mat = ComplexMatrix<Fp>::Zero(U.rows(), U.cols());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate<Fp, Sp> d1 = PauliOperator<Fp, Sp>(targets, mat, controls);
        Gate<Fp, Sp> d2 = PauliOperator<Fp, Sp>(new_targets, mat, {});
        Gate<Fp, Sp> s1 = gate::SparseMatrix<Fp>(targets, mat.sparseView(), controls);
        Gate<Fp, Sp> s2 = gate::SparseMatrix<Fp>(new_targets, mat.sparseView(), {});
        test_gate(d1, d2, n_qubits, control_mask);
        test_gate(s1, s2, n_qubits, control_mask);
    } else if constexpr (num_target == 2) {
        Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U1 =
            get_eigen_matrix_random_one_target_unitary<Fp>();
        Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U2 =
            get_eigen_matrix_random_one_target_unitary<Fp>();
        auto U = internal::kronecker_product<Fp>(U2, U1);
        ComplexMatrix<Fp> mat = ComplexMatrix<Fp>::Zero(U.rows(), U.cols());
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate<Fp, Sp> d1 = PauliOperator<Fp, Sp>(targets, mat, controls);
        Gate<Fp, Sp> d2 = PauliOperator<Fp, Sp>(new_targets, mat, {});
        Gate<Fp, Sp> s1 = gate::SparseMatrix<Fp>(targets, mat.sparseView(), controls);
        Gate<Fp, Sp> s2 = gate::SparseMatrix<Fp>(new_targets, mat.sparseView(), {});
        test_gate<Fp>(d1, d2, n_qubits, control_mask);
        test_gate<Fp>(s1, s2, n_qubits, control_mask);
    } else {
        Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U1 =
            get_eigen_matrix_random_one_target_unitary<Fp>();
        Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U2 =
            get_eigen_matrix_random_one_target_unitary<Fp>();
        Eigen::Matrix<StdComplex<Fp>, 2, 2, Eigen::RowMajor> U3 =
            get_eigen_matrix_random_one_target_unitary<Fp>();
        auto U = internal::kronecker_product<Fp>(U3, internal::kronecker_product<Fp>(U2, U1));
        internal::ComplexMatrix<Fp> mat = ComplexMatrix<Fp>::Zero(U.rows(), U.cols());
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate<Fp, Sp> d1 = PauliOperator<Fp, Sp>(targets, mat, controls);
        Gate<Fp, Sp> d2 = PauliOperator<Fp, Sp>(new_targets, mat, {});
        Gate<Fp, Sp> s1 = gate::SparseMatrix<Fp>(targets, mat.sparseView(), controls);
        Gate<Fp, Sp> s2 = gate::SparseMatrix<Fp>(new_targets, mat.sparseView(), {});
        test_gate<Fp>(d1, d2, n_qubits, control_mask);
        test_gate<Fp>(s1, s2, n_qubits, control_mask);
    }
}

FLOAT_AND_SPACE(Fp, Sp)
void TestControl(std::uint64_t n) {
    for ([[maybe_unused]] std::uint64_t _ : std::views::iota(0, 10)) {
        test_standard_gate_control<Fp, Sp, 0, 1>(gate::GlobalPhase<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::X<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::Y<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::Z<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::S<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::Sdag<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::T<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::Tdag<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::SqrtX<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::SqrtXdag<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::SqrtY<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::SqrtYdag<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::P0<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 0>(gate::P1<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 1>(gate::RX<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 1>(gate::RY<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 1>(gate::RZ<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 1>(gate::U1<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 2>(gate::U2<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 1, 3>(gate::U3<Fp, Sp>, n);
        test_standard_gate_control<Fp, Sp, 2, 0>(gate::Swap<Fp, Sp>, n);
        test_pauli_control<Fp, Sp, false>(n);
        test_pauli_control<Fp, Sp, true>(n);
        test_matrix_control<Fp, Sp, 0>(n);
        test_matrix_control<Fp, Sp, 1>(n);
        test_matrix_control<Fp, Sp, 2>(n);
        test_matrix_control<Fp, Sp, 3>(n);
    }
}
