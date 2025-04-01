#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;
using namespace nlohmann;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        const std::uint64_t n_qubits = 18;
        const std::uint64_t dim = 1 << n_qubits;
        const std::uint64_t scount = 100000;
        std::vector<std::complex<double>> initial_state(dim);
        for (std::uint64_t i = 1; i < dim; i++) {
            initial_state[i] = {static_cast<double>(i + 1), 0.0};
        }
        StateVector<Precision::F64, ExecutionSpace::Default> state(n_qubits);
        state.load(initial_state);

        Circuit<Precision::F64, ExecutionSpace::Default> circuit(2);
        auto i = gate::I<Precision::F64, ExecutionSpace::Default>();
        auto z1 = gate::Z<Precision::F64, ExecutionSpace::Default>(1, {});
        auto z2 = gate::Z<Precision::F64, ExecutionSpace::Default>(0, {1});
        auto p1 = gate::Probablistic<Precision::F64, ExecutionSpace::Default>({0.1, 0.9}, {z1, i});
        auto p2 = gate::Probablistic<Precision::F64, ExecutionSpace::Default>({0.3, 0.7}, {z2, i});
        circuit.add_gate(p1);
        circuit.add_gate(p2);
        circuit.add_gate(p1);
        circuit.add_gate(p2);
        circuit.add_gate(p1);
        circuit.add_gate(p2);
        circuit.add_gate(p1);
        circuit.add_gate(p2);
        circuit.add_gate(p1);
        circuit.add_gate(p2);

        std::cout << "Start Simulate:" << std::endl;
        auto sim = circuit.simulate_noise(state, scount);
        std::cout << "Size:" << sim.size() << std::endl;
        std::uint64_t sum = 0;
        for (auto&& [s, c] : sim) {
            std::cout << "Count:" << c << std::endl;
            sum += c;
        }
        assert(sum == scount);
    }
    scaluq::finalize();
}
