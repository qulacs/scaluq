#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;
using namespace nlohmann;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        StateVector<Precision::F64, ExecutionSpace::Default> state(2);
        state.set_amplitude_at(1, 2);
        state.set_amplitude_at(2, 3);
        state.set_amplitude_at(3, 4);
        Circuit<Precision::F64, ExecutionSpace::Default> circuit(2);
        auto i = gate::I<Precision::F64, ExecutionSpace::Default>();
        auto z1 = gate::Z<Precision::F64, ExecutionSpace::Default>(1, {});
        auto z2 = gate::Z<Precision::F64, ExecutionSpace::Default>(0, {1});
        auto p1 = gate::Probablistic<Precision::F64, ExecutionSpace::Default>({0.1, 0.9}, {z1, i});
        auto p2 = gate::Probablistic<Precision::F64, ExecutionSpace::Default>({0.3, 0.7}, {z2, i});
        circuit.add_gate(p1);
        circuit.add_gate(p2);
        auto v = circuit.simulate_noise(state, 1000);

        std::cout << v.size() << std::endl;

        std::cout << "p1=z1, p2=z2:\n";
        std::cout << "Count: " << v[0].second << std::endl;
        std::cout << v[0].first << "\n";

        std::cout << "p1=z1, p2=i:\n";
        std::cout << "Count: " << v[1].second << std::endl;
        std::cout << v[1].first << "\n";

        std::cout << "p1=i, p2=z2:\n";
        std::cout << "Count: " << v[2].second << std::endl;
        std::cout << v[2].first << "\n";

        std::cout << "p1=i, p2=i:\n";
        std::cout << "Count: " << v[3].second << std::endl;
        std::cout << v[3].first << "\n";
    }
    scaluq::finalize();
}
