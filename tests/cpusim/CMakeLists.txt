cmake_minimum_required(VERSION 3.11)

enable_testing()

add_executable(cpusim_test EXCLUDE_FROM_ALL
    gate_test.cpp
    state_vector_test.cpp
)

target_link_libraries(cpusim_test PRIVATE
    core
    cpusim
    GTest::gtest_main
    Eigen3::Eigen
)
target_include_directories(cpusim_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
