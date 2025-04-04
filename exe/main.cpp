#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;
using namespace nlohmann;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        const std::uint64_t n_qubits = 5;
        const std::uint64_t omega = 3;
        std::vector<std::uint64_t> U_control;
        for (int i = 0; i < n_qubits; ++i) {
            if (i != omega) U_control.push_back(i);
        }
        auto U_w = gate::Z<Precision::F64, ExecutionSpace::Default>(n_qubits - 1, U_control);
    }
    scaluq::finalize();
}
