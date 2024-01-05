cmake_minimum_required(VERSION 3.21)

add_library(qulacs)

target_sources(qulacs PRIVATE
    gate/gate_npair_qubit.cpp
    gate/gate_one_control_one_target.cpp
    gate/gate_one_qubit.cpp
    gate/gate_pauli.cpp
    gate/gate_quantum_matrix.cpp
    gate/gate_two_qubit.cpp
    gate/update_ops_dense_matrix.cpp
    gate/update_ops_npair_qubit.cpp
    gate/update_ops_one_control_one_target.cpp
    gate/update_ops_one_qubit.cpp
    gate/update_ops_pauli.cpp
    gate/update_ops_quantum_matrix.cpp
    gate/update_ops_two_qubit.cpp
    state/state_vector.cpp
    util/random.cpp
)
target_link_libraries(qulacs PRIVATE Kokkos::kokkos)
target_include_directories(qulacs PRIVATE ${PROJECT_SOURCE_DIR}/qulacs)
