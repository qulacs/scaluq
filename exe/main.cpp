#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

template <Precision Prec, ExecutionSpace Space>
std::pair<OperatorBatched<Prec, Space>, std::vector<Operator<Prec, Space>>>
generate_random_observable(int n) {
    Random random;
    std::uint64_t batch_size = random.int32() % 5 + 1;
    std::vector<std::vector<PauliOperator<Prec>>> rand_observable;
    std::vector<Operator<Prec, Space>> test_rand_observable;

    for (std::uint64_t b = 0; b < batch_size; ++b) {
        std::vector<PauliOperator<Prec>> ops;
        std::uint64_t term_count = random.int32() % 10 + 1;
        for (std::uint64_t term = 0; term < term_count; ++term) {
            std::vector<std::uint64_t> paulis(n, 0);
            double coef = random.uniform();
            for (std::uint64_t i = 0; i < paulis.size(); ++i) {
                paulis[i] = random.int32() % 4;
            }

            std::string str = "";
            for (std::uint64_t ind = 0; ind < paulis.size(); ind++) {
                std::uint64_t val = paulis[ind];
                if (val != 0) {
                    if (val == 1)
                        str += " X";
                    else if (val == 2)
                        str += " Y";
                    else if (val == 3)
                        str += " Z";
                    str += " " + std::to_string(ind);
                }
            }
            ops.push_back(PauliOperator<Prec>(str.c_str(), coef));
        }
        rand_observable.push_back(ops);
        test_rand_observable.push_back(Operator<Prec, Space>(ops));
    }
    return {OperatorBatched<Prec, Space>(rand_observable), std::move(test_rand_observable)};
}

template <Precision Prec, ExecutionSpace Space>
void test_gate(ParamGate<Prec> gate_control,
               ParamGate<Prec> gate_simple,
               std::uint64_t n_qubits,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               double param) {
    // Generate random state on Host first to avoid potential issues with Random Pool on
    // HostSerialSpace
    StateVector<Prec, ExecutionSpace::Host> state_host =
        StateVector<Prec, ExecutionSpace::Host>::Haar_random_state(n_qubits);
    auto amplitudes = state_host.get_amplitudes();

    StateVector<Prec, Space> state(n_qubits);
    state.load(amplitudes);

    StateVector<Prec, Space> state_controlled(n_qubits - std::popcount(control_mask));
    std::vector<StdComplex> amplitudes_controlled(state_controlled.dim());
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        amplitudes_controlled[i] =
            amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                       control_value_mask];
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(state, param);
    gate_simple->update_quantum_state(state_controlled, param);
    amplitudes = state.get_amplitudes();
    amplitudes_controlled = state_controlled.get_amplitudes();

    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        if (std::abs(amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                                control_value_mask] -
                     amplitudes_controlled[i]) > 1e-10) {
            std::cerr << "Mismatch at index " << i << std::endl;
            std::exit(1);
        }
    }
}

template <Precision Prec, ExecutionSpace Space>
void test() {
    std::uint64_t n = 10;
    PauliOperator<Prec> pauli1, pauli2;
    std::vector<std::uint64_t> controls, control_values;
    std::uint64_t control_mask = 0, control_value_mask = 0;
    std::uint64_t num_control = 0;
    Random random;
    for (std::uint64_t i : std::views::iota(0ULL, n)) {
        std::uint64_t dat = random.int32() % 12;
        if (dat < 4) {
            pauli1.add_single_pauli(i, dat);
            pauli2.add_single_pauli(i - num_control, dat);
        } else if (dat < 8) {
            controls.push_back(i);
            control_mask |= 1ULL << i;
            bool v = random.int64() & 1;
            control_values.push_back(v);
            control_value_mask |= v << i;
            num_control++;
        }
    }
    double param = random.uniform() * std::numbers::pi * 2;
    ParamGate g1 = gate::ParamPauliRotation(pauli1, 1., controls, control_values);
    ParamGate g2 = gate::ParamPauliRotation(pauli2, 1., {}, {});
    test_gate<Prec, Space>(g1, g2, n, control_mask, control_value_mask, param);
}

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    // for (int _ : std::views::iota(0, 20000)) {
    //     test<Precision::F64, ExecutionSpace::Default>();
    // }
    for (int _ : std::views::iota(0, 100)) {
        test<Precision::F64, ExecutionSpace::HostSerialSpace>();
    }
    {
        ComplexMatrix mat(4, 4);
        mat << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0;
        auto gate = gate::DenseMatrix<Precision::F64, ExecutionSpace::Default>({0, 1}, mat, {}, {});
        StateVector<Precision::F64, ExecutionSpace::Default> state(3);
        gate->update_quantum_state(state);
        std::cout << state << std::endl;
    }
    scaluq::finalize();  // must be called last
}
