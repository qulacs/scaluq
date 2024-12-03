cmake_minimum_required(VERSION 3.21)

enable_testing()

add_executable(scaluq_test EXCLUDE_FROM_ALL
    circuit/circuit_test.cpp
    circuit/param_circuit_test.cpp
    gate/gate_test.cpp
    gate/batched_gate_test.cpp
    # gate/merge_test.cpp
    gate/param_gate_test.cpp
    operator/test_pauli_operator.cpp
    operator/test_operator.cpp
    state/state_vector_test.cpp
    state/state_vector_batched_test.cpp
)

target_link_libraries(scaluq_test PUBLIC 
    scaluq
    Kokkos::kokkos
    GTest::gtest_main
)
target_include_directories(scaluq_test PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(scaluq_test PRIVATE ${eigen_SOURCE_DIR})
