cmake_minimum_required(VERSION 3.11)

enable_testing()

add_executable(qulacs2023_test EXCLUDE_FROM_ALL
    dummy_test.cpp
    #gate/gate_test.cpp
    #state/state_vector_test.cpp
)

target_link_libraries(qulacs2023_test PRIVATE
    qulacs2023
    GTest::gtest_main
    Eigen3::Eigen
)

target_link_libraries(qulacs2023_test PRIVATE qulacs2023 Kokkos::kokkos)
target_include_directories(qulacs2023_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
