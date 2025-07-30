#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

template <Precision Prec, ExecutionSpace Space>
void merge_gate_test() {
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 1; ++repeat) {
        auto op1 = generate_random_observable_with_eigen<Prec, Space>(n, random).first;
        auto op2 = generate_random_observable_with_eigen<Prec, Space>(n, random).first;
        auto op = op1 + op2;
        std::cout << "Operator 1: " << op1.to_string() << std::endl;
        std::cout << "Operator 2: " << op2.to_string() << std::endl;
        std::cout << "Operator: " << op.to_string() << std::endl;
        std::cout << "Operator minus: " << (-op1).to_string() << std::endl;
        auto state = StateVector<Prec, Space>::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp2 = op2.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        std::cout << "Expectation value: " << exp << std::endl;
        std::cout << "Expectation value 1: " << exp1 << std::endl;
        std::cout << "Expectation value 2: " << exp2 << std::endl;
        std::cout << "Difference: " << std::abs(exp1 + exp2 - exp) << std::endl;
    }
}

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    merge_gate_test<Precision::F64, ExecutionSpace::Default>();
    scaluq::finalize();  // must be called last
}
