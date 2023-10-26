cmake_minimum_required(VERSION 3.11)

enable_testing()

add_executable(qulacs_test EXCLUDE_FROM_ALL
    #dummy_test.cpp
    #gate/gate_test.cpp
    #state/state_vector_test.cpp
    #operator/test_pauli_operator.cpp
)

target_link_libraries(qulacs_test PRIVATE
    qulacs
    GTest::gtest_main
    Eigen3::Eigen
)

target_link_libraries(qulacs_test PRIVATE qulacs Kokkos::kokkos)
target_include_directories(qulacs_test PRIVATE ${PROJECT_SOURCE_DIR}/qulacs)
