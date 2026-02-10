cmake_minimum_required(VERSION 3.24)

include(GoogleTest)

file(GLOB_RECURSE ALL_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
)

if(NOT SCALUQ_TEST_SOURCES STREQUAL "ALL" AND DEFINED SCALUQ_TEST_SOURCES)
    add_executable(scaluq_test
        ${SCALUQ_TEST_SOURCES}
    )
else()
    add_executable(scaluq_test
        ${ALL_TEST_SOURCES}
    )
endif()

target_link_libraries(scaluq_test PUBLIC 
    scaluq
    Kokkos::kokkos
    GTest::gtest_main
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
target_include_directories(scaluq_test PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(scaluq_test PRIVATE ${eigen_SOURCE_DIR})

gtest_discover_tests(scaluq_test)
