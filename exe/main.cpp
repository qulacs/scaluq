#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

template <Precision Prec, ExecutionSpace Space>
void merge_gate_test() {
    const std::uint64_t n_qubits = 3;
    StateVector<Prec, Space> state_vector(n_qubits);
    state_vector.load([n_qubits] {
        std::vector<StdComplex> tmp(1 << n_qubits);
        for (std::uint64_t i = 0; i < tmp.size(); ++i) tmp[i] = StdComplex(i, 0);
        return tmp;
    }());

    // PauliOperator<Prec, Space> op(0b001, 0b010, StdComplex(2));
    // op.apply_to_state(state_vector);
    // std::vector<StdComplex> expected = {2, 0, -6, -4, 10, 8, -14, -12};
}

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    merge_gate_test<Precision::F64, ExecutionSpace::Default>();
    scaluq::finalize();  // must be called last
}
