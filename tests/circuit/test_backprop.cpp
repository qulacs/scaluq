#include <gtest/gtest.h>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>
#include <scaluq/operator/operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class CircuitBackpropTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(CircuitBackpropTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space>
void backprop_test() {
    const std::uint64_t n = 3;
    const std::uint64_t dim = 1ULL << n;

    std::default_random_engine engine(0);
    std::uniform_int_distribution<std::uint64_t> dist_target(0, n - 1);
    std::uniform_real_distribution<double> dist_param(-M_PI, M_PI);

    Circuit<Prec> circuit(n);

    const double pcoef1 = 0.7;
    const double pcoef2 = 0.2;
    const double pcoef3 = -0.4;
    PauliOperator<Prec> pauli1("X 0 Y 1", pcoef1);
    PauliOperator<Prec> pauli2("Y 1 Z 2", pcoef2);
    PauliOperator<Prec> pauli3("Z 2 X 0", pcoef3);
    Operator<Prec, Space> op({pauli1, pauli2, pauli3});

    auto X0 = get_expanded_eigen_matrix_with_identity(0, make_X(), n);
    auto Y1 = get_expanded_eigen_matrix_with_identity(1, make_Y(), n);
    auto Z2 = get_expanded_eigen_matrix_with_identity(2, make_Z(), n);
    auto OperatorMatrix = pcoef1 * (X0 * Y1) + pcoef2 * (Y1 * Z2) + pcoef3 * (Z2 * X0);

    std::map<std::string, double> parameters;
    std::vector<std::uint64_t> target_indices;
    for (auto idx = std::size_t{0}; idx < n; idx++) {
        parameters[std::to_string(3 * idx)] = dist_param(engine);
        parameters[std::to_string(3 * idx + 1)] = dist_param(engine);
        parameters[std::to_string(3 * idx + 2)] = dist_param(engine);
        target_indices.push_back(dist_target(engine));
        target_indices.push_back(dist_target(engine));
        target_indices.push_back(dist_target(engine));

        circuit.add_param_gate(gate::ParamRX<Prec>(target_indices[3 * idx]),
                               std::to_string(3 * idx));
        circuit.add_param_gate(gate::ParamRY<Prec>(target_indices[3 * idx + 1]),
                               std::to_string(3 * idx + 1));
        circuit.add_param_gate(gate::ParamRZ<Prec>(target_indices[3 * idx + 2]),
                               std::to_string(3 * idx + 2));
    }
    auto gradients = circuit.backprop(op, parameters);

    // make gradients by eigen matrix calculation
    // make forward state
    ComplexVector state_eigen = ComplexVector::Zero(dim);
    state_eigen[0] = StdComplex(1.0, 0.0);
    std::vector<ComplexMatrix> matrices;
    for (auto idx = std::size_t{0}; idx < n; idx++) {
        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            target_indices[3 * idx], make_RX(parameters.at(std::to_string(3 * idx))), n));
        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            target_indices[3 * idx + 1], make_RY(parameters.at(std::to_string(3 * idx + 1))), n));
        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            target_indices[3 * idx + 2], make_RZ(parameters.at(std::to_string(3 * idx + 2))), n));

        state_eigen = matrices[3 * idx] * state_eigen;
        state_eigen = matrices[3 * idx + 1] * state_eigen;
        state_eigen = matrices[3 * idx + 2] * state_eigen;
    }
    ComplexVector bistate_eigen = OperatorMatrix * state_eigen;

    std::unordered_map<std::string, double> gradients_eigen;

    // apply inverse gates
    for (int idx = matrices.size() - 1; idx >= 0; idx--) {
        auto tmp_state = state_eigen;
        if (idx % 3 == 0) {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            target_indices[idx], make_RX(-M_PI / 1.0), n) *
                        tmp_state;
        } else if (idx % 3 == 1) {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            target_indices[idx], make_RY(-M_PI / 1.0), n) *
                        tmp_state;
        } else {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            target_indices[idx], make_RZ(-M_PI / 1.0), n) *
                        tmp_state;
        }

        const StdComplex ip = (bistate_eigen.adjoint() * tmp_state)(0, 0);
        const double contrib =
            1.0 * static_cast<double>(ip.real()) * 1.0;  // 各gateの係数は1.0としたため
        gradients_eigen[std::to_string(idx)] += contrib;

        const auto& gate = matrices[idx];
        state_eigen = gate.adjoint() * state_eigen;
        bistate_eigen = gate.adjoint() * bistate_eigen;
    }

    for (auto idx = std::size_t{0}; idx < 3 * n; idx++) {
        check_near<Prec>(gradients[std::to_string(idx)], gradients_eigen[std::to_string(idx)]);
    }
}

TYPED_TEST(CircuitBackpropTest, BackpropCircuit) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    backprop_test<Prec, Space>();
}
