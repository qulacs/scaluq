cmake_minimum_required(VERSION 3.21)

enable_testing()

add_executable(qulacs_test EXCLUDE_FROM_ALL
    circuit/circuit_test.cpp
    gate/gate_test.cpp
    operator/test_pauli_operator.cpp
    operator/test_operator.cpp
    state/state_vector_test.cpp
)

target_link_libraries(qulacs_test PRIVATE
    qulacs
    Kokkos::kokkos
    GTest::gtest_main
    Eigen3::Eigen
)

target_include_directories(qulacs_test PRIVATE ${PROJECT_SOURCE_DIR}/qulacs)
