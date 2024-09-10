cmake_minimum_required(VERSION 3.21)

enable_testing()

add_executable(scaluq_test EXCLUDE_FROM_ALL
    # circuit/circuit_test.cpp
    # circuit/param_circuit_test.cpp
    # gate/gate_test.cpp
    # # gate/merge_test.cpp
    # gate/param_gate_test.cpp
    # operator/test_pauli_operator.cpp
    # operator/test_operator.cpp
    state/state_vector_test.cpp
    state/state_vector_batched_test.cpp
)

target_link_libraries(scaluq_test PRIVATE
    scaluq
    Kokkos::kokkos
    GTest::gtest_main
    Eigen3::Eigen
)

target_include_directories(scaluq_test PRIVATE ${PROJECT_SOURCE_DIR}/scaluq)
