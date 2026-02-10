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
void backprop_test_parametric_rc() {
    const std::uint64_t n = 5;
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
    auto operator_matrix = pcoef1 * (X0 * Y1) + pcoef2 * (Y1 * Z2) + pcoef3 * (Z2 * X0);

    std::map<std::string, double> parameters;
    std::vector<std::uint64_t> gate_targets;
    for (auto idx = std::size_t{0}; idx < n; idx++) {
        int idx_first = 3 * idx;
        int idx_second = 3 * idx + 1;
        int idx_third = 3 * idx + 2;

        parameters[std::to_string(idx_first)] = dist_param(engine);
        parameters[std::to_string(idx_second)] = dist_param(engine);
        parameters[std::to_string(idx_third)] = dist_param(engine);
        gate_targets.push_back(dist_target(engine));
        gate_targets.push_back(dist_target(engine));
        gate_targets.push_back(dist_target(engine));

        circuit.add_param_gate(gate::ParamRX<Prec>(gate_targets[idx_first]),
                               std::to_string(idx_first));
        circuit.add_param_gate(gate::ParamRY<Prec>(gate_targets[idx_second]),
                               std::to_string(idx_second));
        circuit.add_param_gate(gate::ParamRZ<Prec>(gate_targets[idx_third]),
                               std::to_string(idx_third));
    }
    auto gradients = circuit.backprop(op, parameters);

    // make gradients by eigen matrix calculation
    // make forward state
    ComplexVector state_eigen = ComplexVector::Zero(dim);
    state_eigen[0] = StdComplex(1.0, 0.0);
    std::vector<ComplexMatrix> matrices;
    for (auto idx = std::size_t{0}; idx < n; idx++) {
        const std::size_t idx_first = 3 * idx;
        const std::size_t idx_second = 3 * idx + 1;
        const std::size_t idx_third = 3 * idx + 2;

        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            gate_targets[idx_first], make_RX(parameters.at(std::to_string(idx_first))), n));
        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            gate_targets[idx_second], make_RY(parameters.at(std::to_string(idx_second))), n));
        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            gate_targets[idx_third], make_RZ(parameters.at(std::to_string(idx_third))), n));

        state_eigen = matrices[idx_first] * state_eigen;
        state_eigen = matrices[idx_second] * state_eigen;
        state_eigen = matrices[idx_third] * state_eigen;
    }
    ComplexVector bistate_eigen = operator_matrix * state_eigen;

    std::unordered_map<std::string, double> gradients_eigen;

    // apply inverse gates
    for (int idx = matrices.size() - 1; idx >= 0; idx--) {
        auto tmp_state = state_eigen;
        if (idx % 3 == 0) {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            gate_targets[idx], make_RX(-M_PI / 1.0), n) *
                        tmp_state;
        } else if (idx % 3 == 1) {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            gate_targets[idx], make_RY(-M_PI / 1.0), n) *
                        tmp_state;
        } else {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            gate_targets[idx], make_RZ(-M_PI / 1.0), n) *
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

template <Precision Prec, ExecutionSpace Space>
void backprop_test_parametric_rotation() {
    const std::uint64_t n = 5;
    const std::uint64_t dim = 1ULL << n;

    std::default_random_engine engine(0);
    std::uniform_int_distribution<std::uint64_t> dist_target(0, n - 1);
    std::uniform_real_distribution<double> dist_param(-M_PI, M_PI);

    Circuit<Prec> circuit(n);

    const double pcoef1 = 0.7;
    const double pcoef2 = 0.2;
    const double pcoef3 = -0.4;
    const PauliOperator<Prec> pauli1("X 0 Y 1", pcoef1);
    const PauliOperator<Prec> pauli2("Y 1 Z 2", pcoef2);
    const PauliOperator<Prec> pauli3("Z 2 X 0", pcoef3);
    Operator<Prec, Space> op({pauli1, pauli2, pauli3});

    const auto X0 = get_expanded_eigen_matrix_with_identity(0, make_X(), n);
    const auto Y1 = get_expanded_eigen_matrix_with_identity(1, make_Y(), n);
    const auto Z2 = get_expanded_eigen_matrix_with_identity(2, make_Z(), n);
    const auto operator_matrix = pcoef1 * (X0 * Y1) + pcoef2 * (Y1 * Z2) + pcoef3 * (Z2 * X0);

    std::map<std::string, double> parameters;
    std::vector<double> pauli_rotation_coefs;
    std::vector<double> pauli_op_coefs;
    std::vector<std::uint64_t> gate_targets;
    for (auto idx = std::size_t{0}; idx < n; idx++) {
        const std::size_t idx_first = 3 * idx;
        const std::size_t idx_second = 3 * idx + 1;
        const std::size_t idx_third = 3 * idx + 2;

        parameters[std::to_string(idx_first)] = dist_param(engine);
        parameters[std::to_string(idx_second)] = dist_param(engine);
        parameters[std::to_string(idx_third)] = dist_param(engine);
        gate_targets.push_back(dist_target(engine));
        gate_targets.push_back(dist_target(engine));
        gate_targets.push_back(dist_target(engine));
        pauli_rotation_coefs.push_back(dist_param(engine));
        pauli_rotation_coefs.push_back(dist_param(engine));
        pauli_rotation_coefs.push_back(dist_param(engine));
        pauli_op_coefs.push_back(dist_param(engine));
        pauli_op_coefs.push_back(dist_param(engine));
        pauli_op_coefs.push_back(dist_param(engine));

        circuit.add_param_gate(
            gate::ParamPauliRotation<Prec>(
                PauliOperator<Prec>("X " + std::to_string(gate_targets[idx_first]),
                                    pauli_op_coefs[idx_first]),
                pauli_rotation_coefs[idx_first]),
            std::to_string(idx_first));
        circuit.add_param_gate(
            gate::ParamPauliRotation<Prec>(
                PauliOperator<Prec>("Y " + std::to_string(gate_targets[idx_second]),
                                    pauli_op_coefs[idx_second]),
                pauli_rotation_coefs[idx_second]),
            std::to_string(idx_second));
        circuit.add_param_gate(
            gate::ParamPauliRotation<Prec>(
                PauliOperator<Prec>("Z " + std::to_string(gate_targets[idx_third]),
                                    pauli_op_coefs[idx_third]),
                pauli_rotation_coefs[idx_third]),
            std::to_string(idx_third));
    }
    auto gradients = circuit.backprop(op, parameters);

    // make gradients by eigen matrix calculation
    // make forward state
    ComplexVector state_eigen = ComplexVector::Zero(dim);
    state_eigen[0] = StdComplex(1.0, 0.0);
    std::vector<ComplexMatrix> matrices;
    const auto make_angle_rc = [&](int idx) {
        return parameters.at(std::to_string(idx)) * pauli_op_coefs[idx] * pauli_rotation_coefs[idx];
    };
    for (auto idx = std::size_t{0}; idx < n; idx++) {
        const std::size_t idx_first = 3 * idx;
        const std::size_t idx_second = 3 * idx + 1;
        const std::size_t idx_third = 3 * idx + 2;

        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            gate_targets[idx_first], make_RX(make_angle_rc(idx_first)), n));
        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            gate_targets[idx_second], make_RY(make_angle_rc(idx_second)), n));
        matrices.push_back(get_expanded_eigen_matrix_with_identity(
            gate_targets[idx_third], make_RZ(make_angle_rc(idx_third)), n));

        state_eigen = matrices[idx_first] * state_eigen;
        state_eigen = matrices[idx_second] * state_eigen;
        state_eigen = matrices[idx_third] * state_eigen;
    }
    ComplexVector bistate_eigen = operator_matrix * state_eigen;

    std::unordered_map<std::string, double> gradients_eigen;

    // apply inverse gates
    const auto make_scale_rc = [&](int idx) {
        return pauli_op_coefs[idx] * pauli_rotation_coefs[idx];
    };
    for (int idx = matrices.size() - 1; idx >= 0; idx--) {
        auto tmp_state = state_eigen;
        if (idx % 3 == 0) {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            gate_targets[idx], make_RX(-M_PI / 1.0), n) *
                        tmp_state;
        } else if (idx % 3 == 1) {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            gate_targets[idx], make_RY(-M_PI / 1.0), n) *
                        tmp_state;
        } else {
            tmp_state = get_expanded_eigen_matrix_with_identity(
                            gate_targets[idx], make_RZ(-M_PI / 1.0), n) *
                        tmp_state;
        }

        const StdComplex ip = (bistate_eigen.adjoint() * tmp_state)(0, 0);
        const double contrib =
            make_scale_rc(idx) * static_cast<double>(ip.real()) * make_scale_rc(idx);
        gradients_eigen[std::to_string(idx)] += contrib;

        const auto& gate = matrices[idx];
        state_eigen = gate.adjoint() * state_eigen;
        bistate_eigen = gate.adjoint() * bistate_eigen;
    }

    for (auto idx = std::size_t{0}; idx < 3 * n; idx++) {
        check_near<Prec>(gradients[std::to_string(idx)], gradients_eigen[std::to_string(idx)]);
    }
}

TYPED_TEST(CircuitBackpropTest, BackpropCircuitParametricRC) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    backprop_test_parametric_rc<Prec, Space>();
}

TYPED_TEST(CircuitBackpropTest, BackpropCircuitPauliRotation) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    backprop_test_parametric_rotation<Prec, Space>();
}
