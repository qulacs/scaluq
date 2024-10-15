cmake_minimum_required(VERSION 3.21)

target_sources(scaluq PRIVATE
    circuit/circuit.cpp
    gate/update_ops_dense_matrix.cpp
    gate/update_ops_standard.cpp
    # # gate/merge_gate.cpp
    operator/apply_pauli.cpp
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
