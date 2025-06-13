cmake_minimum_required(VERSION 3.24)

function(exe name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        scaluq
        Eigen3::Eigen
        Kokkos::kokkos
        nlohmann_json::nlohmann_json
    )
    if (SCALUQ_USE_TEST)
        target_link_libraries(${name} PRIVATE GTest::gtest_main)
    endif(SCALUQ_USE_TEST)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
endfunction()

exe(main)
