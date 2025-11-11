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
    std::vector<std::vector<PauliOperator<Prec, Space>>> rand_observable;
    std::vector<Operator<Prec, Space>> test_rand_observable;

    for (std::uint64_t b = 0; b < batch_size; ++b) {
        std::vector<PauliOperator<Prec, Space>> ops;
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
            ops.push_back(PauliOperator<Prec, Space>(str.c_str(), coef));
        }
        rand_observable.push_back(ops);
        test_rand_observable.push_back(Operator<Prec, Space>(ops));
    }
    return {OperatorBatched<Prec, Space>(rand_observable), std::move(test_rand_observable)};
}

template <Precision Prec, ExecutionSpace Space>
void merge_gate_test() {
    StateVector<Prec, Space> state_vector(6);
    state_vector.set_Haar_random_state();
    StateVector<Prec, Space> state_vector_trans(6);
    state_vector.set_Haar_random_state();
    {
        std::vector<std::vector<PauliOperator<Prec, Space>>> ops(3);
        ops[0].push_back({PauliOperator<Prec, Space>("X 0 Y 1")});
        ops[0].push_back({PauliOperator<Prec, Space>("Z 2 X 3", 2.0)});
        ops[0].push_back({PauliOperator<Prec, Space>("Y 4 Z 5")});
        ops[1].push_back({PauliOperator<Prec, Space>("X 2 Y 1")});
        ops[1].push_back({PauliOperator<Prec, Space>("Z 2 X 3", 2.0)});
        ops[1].push_back({PauliOperator<Prec, Space>("Y 4 Z 5")});
        ops[2].push_back({PauliOperator<Prec, Space>("X 0 Y 1")});
        ops[2].push_back({PauliOperator<Prec, Space>("Z 2 X 3", 2.0)});
        OperatorBatched<Prec, Space> op_batched(ops);
        auto res = op_batched.get_expectation_value(state_vector);
        std::cout << "Expectation values: ";
        for (const auto& val : res) {
            std::cout << val << " ";
        }
        auto res_trans = op_batched.get_transition_amplitude(state_vector, state_vector_trans);
        std::cout << "Transition values: ";
        for (const auto& val : res) {
            std::cout << val << " ";
        }
        auto op_b_cpy = op_batched.copy();
    }
    {
        Operator<Prec, Space> op1({PauliOperator<Prec, Space>("X 0 Y 1"),
                                   PauliOperator<Prec, Space>("Z 2 X 3", 2.0),
                                   PauliOperator<Prec, Space>("Y 4 Z 5")});
        Operator<Prec, Space> op2({PauliOperator<Prec, Space>("X 2 Y 1"),
                                   PauliOperator<Prec, Space>("Z 2 X 3", 2.0),
                                   PauliOperator<Prec, Space>("Y 4 Z 5")});
        Operator<Prec, Space> op3(
            {PauliOperator<Prec, Space>("X 0 Y 1"), PauliOperator<Prec, Space>("Z 2 X 3", 2.0)});
        std::cout << "Expectation value of op1: " << op1.get_expectation_value(state_vector)
                  << std::endl;
        std::cout << "Expectation value of op2: " << op2.get_expectation_value(state_vector)
                  << std::endl;
        std::cout << "Expectation value of op3: " << op3.get_expectation_value(state_vector)
                  << std::endl;
        std::cout << "Transition value of op1: "
                  << op1.get_transition_amplitude(state_vector, state_vector) << std::endl;
        std::cout << "Transition value of op2: "
                  << op2.get_transition_amplitude(state_vector, state_vector) << std::endl;
        std::cout << "Transition value of op3: "
                  << op3.get_transition_amplitude(state_vector, state_vector) << std::endl;
        std::cout << "operator: " << op1 << std::endl;
        std::cout << "operator: " << op2 << std::endl;
        std::cout << "operator: " << op3 << std::endl;
    }
}

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    merge_gate_test<Precision::F64, ExecutionSpace::Default>();
    scaluq::finalize();  // must be called last
}
