cmake_minimum_required(VERSION 3.21)

target_sources(scaluq PRIVATE
    circuit/circuit.cpp
    gate/update_ops_dense_matrix.cpp
    gate/update_ops_one_qubit.cpp
    gate/update_ops_pauli.cpp
    gate/update_ops_quantum_matrix.cpp
    gate/update_ops_two_qubit.cpp
    gate/update_ops_zero_qubit.cpp
    operator/pauli_operator.cpp
    operator/operator.cpp
    state/state_vector.cpp
    state/state_vector_batched.cpp
)
target_link_libraries(scaluq PRIVATE 
    Kokkos::kokkos
    Eigen3::Eigen
)
target_include_directories(scaluq PRIVATE ${PROJECT_SOURCE_DIR}/scaluq)
