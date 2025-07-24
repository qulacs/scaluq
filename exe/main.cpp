#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

template <Precision Prec, ExecutionSpace Space>
void merge_gate_test() {
    Kokkos::View<PauliOperator<Prec, Space>*> pauli_ops("pauli_ops", 3);
    Kokkos::parallel_for(
        "initialize_pauli_ops",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, 3),
        KOKKOS_LAMBDA(std::uint64_t i) {
            if (i == 0) {
                pauli_ops(i) = PauliOperator<Prec, Space>("X 0 Y 1", 1.);
            } else if (i == 1) {
                pauli_ops(i) = PauliOperator<Prec, Space>("Y 0 Z 1", 2.);
            } else {
                pauli_ops(i) = PauliOperator<Prec, Space>("Z 0 X 1", 3.);
            }
            pauli_ops(i).set_coef(1.0);
            auto c = pauli_ops(i).coef();
        });
}

int main() {
    scaluq::initialize();  // 初期化

    merge_gate_test<Precision::F64, ExecutionSpace::Default>();

    scaluq::finalize();  // 終了処理
}
