#include <cstdint>
#include <iostream>
// #include <scaluq/all.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/util/utility.hpp>
using namespace scaluq;
using namespace nlohmann;

int main() {
    constexpr Precision Prec = Precision::F64;
    scaluq::initialize();  // must be called before using any scaluq methods

    {
        Kokkos::Timer tm;
        std::uint64_t n_qubits = 28;
        scaluq::StateVector<scaluq::Precision::F64, scaluq::DefaultSpace> gpu_state(n_qubits);

        for (int i = 0; i < 10; ++i) {
            // auto xgate = scaluq::gate::X<scaluq::Precision::F64, scaluq::DefaultSpace>(2);
            // xgate->update_quantum_state(gpu_state);
        }
        std::cout << tm.seconds() << std::endl;
    }
    {
        Kokkos::Timer tm;
        std::uint64_t n_qubits = 28;
        scaluq::StateVector<scaluq::Precision::F64, scaluq::HostSpace> cpu_state(n_qubits);

        for (int i = 0; i < 10; ++i) {
            // auto xgate = scaluq::gate::X<scaluq::Precision::F64, scaluq::HostSpace>(2);
            // xgate->update_quantum_state(cpu_state);
        }
        std::cout << tm.seconds() << std::endl;
    }
    {
        Kokkos::Timer tm;
        std::uint64_t n_qubits = 28;
        scaluq::StateVector<scaluq::Precision::F32, scaluq::DefaultSpace> gpu_state(n_qubits);

        for (int i = 0; i < 10; ++i) {
            // auto xgate = scaluq::gate::X<scaluq::Precision::F32, DefaultSpace>(2);
            // xgate->update_quantum_state(gpu_state);
        }
        std::cout << tm.seconds() << std::endl;
    }
    // {
    //     Kokkos::Timer tm;
    //     std::uint64_t n_qubits = 28;
    //     scaluq::StateVector<scaluq::Precision::F32, HostSpace> cpu_state(n_qubits);

    //     for (int i = 0; i < 10; ++i) {
    //         auto xgate = scaluq::gate::X<scaluq::Precision::F32, HostSpace>(2);
    //         xgate->update_quantum_state(cpu_state);
    //     }
    //     std::cout << tm.seconds() << std::endl;
    // }

    // {
    //     Kokkos::Timer tm;
    //     std::uint64_t n_qubits = 5;
    //     scaluq::StateVector<scaluq::Precision::F64, DefaultSpace> gpu_state(n_qubits);

    //     for (int i = 0; i < 1000; ++i) {
    //         auto xgate = scaluq::gate::X<scaluq::Precision::F64, DefaultSpace>(2);
    //         xgate->update_quantum_state(gpu_state);
    //     }
    //     std::cout << tm.seconds() << std::endl;
    // }
    // {
    //     Kokkos::Timer tm;
    //     std::uint64_t n_qubits = 5;
    //     scaluq::StateVector<scaluq::Precision::F64, HostSpace> cpu_state(n_qubits);

    //     for (int i = 0; i < 1000; ++i) {
    //         auto xgate = scaluq::gate::X<scaluq::Precision::F64, HostSpace>(2);
    //         xgate->update_quantum_state(cpu_state);
    //     }
    //     std::cout << tm.seconds() << std::endl;
    // }
    // {
    //     Kokkos::Timer tm;
    //     std::uint64_t n_qubits = 5;
    //     scaluq::StateVector<scaluq::Precision::F32, DefaultSpace> gpu_state(n_qubits);

    //     for (int i = 0; i < 1000; ++i) {
    //         auto xgate = scaluq::gate::X<scaluq::Precision::F32, DefaultSpace>(2);
    //         xgate->update_quantum_state(gpu_state);
    //     }
    //     std::cout << tm.seconds() << std::endl;
    // }
    // {
    //     std::uint64_t n_qubits = 5;
    //     scaluq::StateVector<scaluq::Precision::F32, HostSpace> cpu_state(n_qubits);

    //     Kokkos::Timer tm;
    //     for (int i = 0; i < 1000; ++i) {
    //         auto xgate = scaluq::gate::X<scaluq::Precision::F32, HostSpace>(2);
    //         xgate->update_quantum_state(cpu_state);
    //     }
    //     std::cout << tm.seconds() << std::endl;
    // }

    // // {
    // //     std::uint64_t n_qubits = 3;
    // //     scaluq::StateVector<scaluq::Precision::F64, scaluq::DefaultSpace> state(n_qubits);
    // //     state.load({0, 1, 2, 3, 4, 5, 6, 7});
    // //     Json j = state;
    // //     std::cout << j << std::endl;
    // //     state = j;
    // //     std::cout << state << std::endl;

    // //     scaluq::Circuit<scaluq::Precision::F64, scaluq::DefaultSpace> circuit(n_qubits);
    // //     circuit.add_gate(scaluq::gate::X<scaluq::Precision::F64, scaluq::DefaultSpace>(0));
    // //     circuit.add_gate(scaluq::gate::CNot<scaluq::Precision::F64, scaluq::DefaultSpace>(0,
    // 1));
    // //     circuit.add_gate(scaluq::gate::Y<scaluq::Precision::F64, scaluq::DefaultSpace>(1));
    // //     circuit.add_gate(scaluq::gate::RX<scaluq::Precision::F64, scaluq::DefaultSpace>(
    // //         1, std::numbers::pi / 2));
    // //     circuit.update_quantum_state(state);

    // //     scaluq::Operator<scaluq::Precision::F64, scaluq::DefaultSpace> observable(n_qubits);
    // //     observable.add_random_operator(1, 0);
    // //     auto value = observable.get_expectation_value(state);
    // //     std::cout << value << std::endl;
    // // }
    // {
    //     std::uint64_t n_qubits = 2, batch_size = 2;
    //     scaluq::StateVectorBatched<scaluq::Precision::F64, scaluq::DefaultSpace>
    //     states(batch_size,
    //                                                                                     n_qubits);
    //     states.set_Haar_random_state(batch_size, n_qubits, false);
    //     Json j = states;
    //     std::cout << j << std::endl;
    //     states = j;
    //     std::cout << states << std::endl;
    // }
    // {
    //     double coef = 2.0;
    //     std::string pauli_string = "X 0 Z 1 Y 2";
    //     PauliOperator<scaluq::Precision::F64, scaluq::DefaultSpace> pauli(pauli_string, coef);
    //     Json j = pauli;
    //     std::cout << j << std::endl;
    //     pauli = j;
    // }
    // {
    //     std::uint64_t n_qubits = 3;
    //     Operator<scaluq::Precision::F64, scaluq::DefaultSpace> op(n_qubits);
    //     op.add_operator({0b001, 0b010, StdComplex(2)});
    //     op.add_operator({"X 2 Y 1", 1});
    //     Json j = op;
    //     std::cout << j << std::endl;
    //     op = j;
    // }
    // {
    //     std::cout << Json(gate::I<scaluq::Precision::F64, scaluq::DefaultSpace>()) << std::endl;
    //     std::cout << Json(gate::X<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::Y<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::Z<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::H<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::S<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::Sdag<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::T<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::Tdag<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::SqrtX<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::SqrtXdag<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0,
    //     3}))
    //               << std::endl;
    //     std::cout << Json(gate::SqrtY<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::SqrtYdag<scaluq::Precision::F64, scaluq::DefaultSpace>(2, {0,
    //     3}))
    //               << std::endl;
    //     std::cout << Json(gate::RX<scaluq::Precision::F64, scaluq::DefaultSpace>(2, 0.5, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::RY<scaluq::Precision::F64, scaluq::DefaultSpace>(2, 0.5, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::RZ<scaluq::Precision::F64, scaluq::DefaultSpace>(2, 0.5, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::U1<scaluq::Precision::F64, scaluq::DefaultSpace>(2, 0.5, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::U2<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //                      2, 0.5, 0.3, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::U3<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //                      2, 0.5, 0.3, 0.1, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::Swap<scaluq::Precision::F64, scaluq::DefaultSpace>(1, 2, {0, 3}))
    //               << std::endl;

    //     PauliOperator<scaluq::Precision::F64, scaluq::DefaultSpace> pauli("X 2 Y 1");
    //     std::cout << Json(gate::Pauli<scaluq::Precision::F64, scaluq::DefaultSpace>(pauli))
    //               << std::endl;
    //     std::cout << Json(gate::PauliRotation<scaluq::Precision::F64,
    //     scaluq::DefaultSpace>(pauli,
    //                                                                                         0.5))
    //               << std::endl;

    //     auto probgate = gate::Probablistic<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //         {.1, .9},
    //         {gate::X<scaluq::Precision::F64, scaluq::DefaultSpace>(0),
    //          gate::I<scaluq::Precision::F64, scaluq::DefaultSpace>()});
    //     std::cout << Json(probgate) << std::endl;

    //     std::cout << Json(gate::ParamRX<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //                      2, 1.5, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::ParamRY<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //                      2, 1.5, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::ParamRZ<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //                      2, 1.5, {0, 3}))
    //               << std::endl;
    //     std::cout << Json(gate::ParamPauliRotation<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //                      pauli, 0.5))
    //               << std::endl;

    //     auto paramprobgate = gate::ParamProbablistic<scaluq::Precision::F64,
    //     scaluq::DefaultSpace>(
    //         {.1, .9},
    //         {gate::ParamRX<scaluq::Precision::F64, scaluq::DefaultSpace>(0),
    //          gate::I<scaluq::Precision::F64, scaluq::DefaultSpace>()});
    //     std::cout << Json(paramprobgate) << std::endl;
    // }

    // {
    //     auto x = gate::X<scaluq::Precision::F64, scaluq::DefaultSpace>(1, {2});
    //     Json j = x;
    //     std::cout << j << std::endl;
    //     Gate<scaluq::Precision::F64, scaluq::DefaultSpace> gate = j;
    //     std::cout << gate << std::endl;
    // }
    // {
    //     auto x = gate::RX<scaluq::Precision::F64, scaluq::DefaultSpace>(1, 0.5, {2});
    //     Json j = x;
    //     std::cout << j << std::endl;
    //     Gate<scaluq::Precision::F64, scaluq::DefaultSpace> gate = j;
    //     std::cout << gate << std::endl;
    // }
    // {
    //     auto x = gate::Swap<scaluq::Precision::F64, scaluq::DefaultSpace>(1, 3, {2});
    //     Json j = x;
    //     std::cout << j << std::endl;
    //     Gate<scaluq::Precision::F64, scaluq::DefaultSpace> gate = j;
    //     std::cout << gate << std::endl;
    // }
    // {
    //     PauliOperator<scaluq::Precision::F64, scaluq::DefaultSpace> pauli("X 2 Y 1");
    //     auto x =
    //         gate::PauliRotation<scaluq::Precision::F64, scaluq::DefaultSpace>(pauli, 1.5, {0,
    //         3});
    //     Json j = x;
    //     std::cout << j << std::endl;
    //     Gate<scaluq::Precision::F64, scaluq::DefaultSpace> gate = j;
    //     std::cout << gate << std::endl;
    // }
    // {
    //     auto probgate = gate::Probablistic<scaluq::Precision::F64, scaluq::DefaultSpace>(
    //         {.1, .9},
    //         {gate::X<scaluq::Precision::F64, scaluq::DefaultSpace>(0, {2, 3}),
    //          gate::I<scaluq::Precision::F64, scaluq::DefaultSpace>()});
    //     Json j = probgate;
    //     std::cout << j << std::endl;
    //     Gate<scaluq::Precision::F64, scaluq::DefaultSpace> gate = j;
    //     std::cout << gate << std::endl;
    // }
    // {
    //     auto x = gate::ParamRX<scaluq::Precision::F64, scaluq::DefaultSpace>(1, {2});
    //     Json j = x;
    //     std::cout << j << std::endl;
    //     ParamGate<scaluq::Precision::F64, scaluq::DefaultSpace> gate = j;
    //     std::cout << gate << std::endl;
    // }
    // {
    //     auto paramprobgate = gate::ParamProbablistic<scaluq::Precision::F64,
    //     scaluq::DefaultSpace>(
    //         {.1, .9},
    //         {gate::ParamRX<scaluq::Precision::F64, scaluq::DefaultSpace>(0),
    //          gate::I<scaluq::Precision::F64, scaluq::DefaultSpace>()});
    //     Json j = paramprobgate;
    //     std::cout << j << std::endl;
    //     ParamGate<scaluq::Precision::F64, scaluq::DefaultSpace> gate = j;
    //     std::cout << gate << std::endl;
    // }

    // {
    //     Circuit<scaluq::Precision::F64> circuit(10);
    //
    // }

    Kokkos::finalize();

    std::cout << "Type of _Float64: " << typeid(_Float64).name() << std::endl;
    std::cout << "Type of scaluq::Precision::F64: " << typeid(scaluq::Precision::F64).name()
              << std::endl;
    std::cout << "Type of _Float32: " << typeid(_Float32).name() << std::endl;
    std::cout << "Type of scaluq::Precision::F32: " << typeid(scaluq::Precision::F32).name()
              << std::endl;
}
