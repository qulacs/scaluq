cmake_minimum_required(VERSION 3.11)

add_library(qulacs)

target_sources(qulacs PRIVATE
    gate/gate_one_control_one_target.cpp
    gate/gate_one_qubit.cpp
    gate/gate_quantum_matrix.cpp
    gate/gate_two_qubit.cpp
    gate/update_ops_matrix_dense_single.cpp
    gate/update_ops_one_qubit.cpp
    gate/update_ops_one_control_one_target.cpp
    gate/update_ops_quantum_matrix.cpp
    gate/update_ops_two_qubit.cpp
    state/state_vector.cpp
    util/random.cpp
)
target_link_libraries(qulacs PRIVATE Kokkos::kokkos)
target_include_directories(qulacs PRIVATE ${PROJECT_SOURCE_DIR}/qulacs)
