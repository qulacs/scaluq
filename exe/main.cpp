#include <Eigen/Core>
#include <functional>
#include <iostream>

#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

// obtain single dense matrix
inline Eigen::MatrixXcd get_eigen_matrix_single_Pauli(UINT pauli_id) {
    Eigen::MatrixXcd mat(2, 2);
    if (pauli_id == 0)
        mat << 1, 0, 0, 1;
    else if (pauli_id == 1)
        mat << 0, 1, 1, 0;
    else if (pauli_id == 2)
        mat << 0, -1.i, 1.i, 0;
    else if (pauli_id == 3)
        mat << 1, 0, 0, -1;
    return mat;
}
inline Eigen::MatrixXcd get_eigen_matrix_random_single_qubit_unitary() {
    Eigen::MatrixXcd Identity, X, Y, Z;
    Identity = get_eigen_matrix_single_Pauli(0);
    X = get_eigen_matrix_single_Pauli(1);
    Y = get_eigen_matrix_single_Pauli(2);
    Z = get_eigen_matrix_single_Pauli(3);

    double icoef, xcoef, ycoef, zcoef, norm;
    Random random;
    icoef = random.uniform();
    xcoef = random.uniform();
    ycoef = random.uniform();
    zcoef = random.uniform();
    norm = sqrt(icoef * icoef + xcoef * xcoef + ycoef * ycoef + zcoef * zcoef);
    icoef /= norm;
    xcoef /= norm;
    ycoef /= norm;
    zcoef /= norm;
    return icoef * Identity + 1.i * xcoef * X + 1.i * ycoef * Y + 1.i * zcoef * Z;
}

inline SparseComplexMatrix make_sparse_complex_matrix(const UINT n_qubit, double border = 0.9) {
    const UINT dim = 1ULL << n_qubit;
    typedef Eigen::Triplet<StdComplex> T;
    std::vector<T> tripletList;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (UINT i = 0; i < dim; ++i) {
        for (UINT j = 0; j < dim; ++j) {
            if (dis(gen) > border) {
                double real = dis(gen);
                double imag = dis(gen);
                tripletList.push_back(T(i, j, StdComplex(real, imag)));
            }
        }
    }

    SparseComplexMatrix mat(dim, dim);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

inline Eigen::MatrixXcd get_expanded_eigen_matrix_with_identity(
    UINT target_qubit_index, const Eigen::MatrixXcd& one_qubit_matrix, UINT qubit_count) {
    const UINT left_dim = 1ULL << target_qubit_index;
    const UINT right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = Eigen::MatrixXcd::Identity(left_dim, left_dim);
    auto right_identity = Eigen::MatrixXcd::Identity(right_dim, right_dim);
    return internal::kronecker_product(
        internal::kronecker_product(right_identity, one_qubit_matrix), left_identity);
}

string bin(UINT x, UINT n_qubits) {
    string s = "";
    for (UINT i = 0; i < n_qubits; i++) {
        if (x & (1 << i)) {
            s += "1";
        } else {
            s += "0";
        }
    }
    reverse(s.begin(), s.end());
    return s;
}

double norm(std::complex<double> val) {
    return std::sqrt(val.real() * val.real() + val.imag() * val.imag());
}

void run() {
    UINT n_qubits = 4;
    vector<UINT> targets = {2, 1, 3};
    vector<UINT> controls = {};
    StateVector state = StateVector::Haar_random_state(n_qubits);
    UINT dim = 1ULL << n_qubits;
    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> u1, u2, u3;
    Eigen::Matrix<StdComplex, 8, 8, Eigen::RowMajor> Umerge;
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    auto state_cp = state.amplitudes();
    for (UINT i = 0; i < dim; i++) {
        test_state[i] = state_cp[i];
    }
    u1 = make_sparse_complex_matrix(1, 0.2);
    u2 = make_sparse_complex_matrix(1, 0.2);
    u3 = make_sparse_complex_matrix(1, 0.2);
    // u1 = get_eigen_matrix_random_single_qubit_unitary();
    // u2 = get_eigen_matrix_random_single_qubit_unitary();
    // u3 = get_eigen_matrix_random_single_qubit_unitary();
    Umerge = internal::kronecker_product(u3, internal::kronecker_product(u2, u1));
    test_state = get_expanded_eigen_matrix_with_identity(targets[2], u3, n_qubits) *
                 get_expanded_eigen_matrix_with_identity(targets[1], u2, n_qubits) *
                 get_expanded_eigen_matrix_with_identity(targets[0], u1, n_qubits) * test_state;
    SparseComplexMatrix mat = Umerge.sparseView();
    Gate sparse_gate = gate::SparseMatrix(mat, targets, controls);
    sparse_gate->update_quantum_state(state);
    // ComplexMatrix mat = Umerge;
    // Gate dense_gate = gate::DenseMatrix(mat, targets, controls);
    // dense_gate->update_quantum_state(state);
    state_cp = state.amplitudes();
    for (UINT i = 0; i < dim; i++) {
        double diff = norm(test_state[i]) - norm(state_cp[i]);
        diff = diff > 0 ? diff : diff * -1.0;
        cout << "index: " << bin(i, n_qubits) << ", diff: " << diff
             << ", test_state: " << test_state[i] << ", state_cp: " << state_cp[i] << endl;
    }
    return;
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
