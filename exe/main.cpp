#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

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
        std::cout << std::endl;
        std::cout << "Batched operator: " << op_batched << std::endl;
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
