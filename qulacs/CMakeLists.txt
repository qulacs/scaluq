cmake_minimum_required(VERSION 3.21)

add_library(qulacs)

target_sources(qulacs PRIVATE
    # gate/gate_one_control_one_target.cpp
    # gate/gate_one_qubit.cpp
    # gate/gate_quantum_matrix.cpp
    # gate/update_ops_matrix_dense_single.cpp
    # gate/update_ops_one_qubit.cpp
    # gate/update_ops_one_target_one_control.cpp
    # gate/update_ops_quantum_matrix.cpp
    state/state_vector.cpp
    util/random.cpp
)
target_link_libraries(qulacs PRIVATE Kokkos::kokkos)
target_include_directories(qulacs PRIVATE ${PROJECT_SOURCE_DIR}/qulacs)
