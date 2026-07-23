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

option(SCALUQ_BUILD_QULACS_SIMD_PATH_BENCHMARK
       "Build the Scaluq/Qulacs 4x4 SIMD path benchmark" OFF)

if(SCALUQ_BUILD_QULACS_SIMD_PATH_BENCHMARK)
    include(FetchContent)
    FetchContent_Declare(
        qulacs_simd_path_benchmark
        GIT_REPOSITORY https://github.com/qulacs/qulacs.git
        GIT_TAG a8b0b46604571968b4c0bb80bce91ea9590eb9fd # v0.6.12
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(qulacs_simd_path_benchmark)
    if(NOT qulacs_simd_path_benchmark_POPULATED)
        FetchContent_Populate(qulacs_simd_path_benchmark)
    endif()

    add_library(qulacs_simd_path_kernels STATIC
        ${qulacs_simd_path_benchmark_SOURCE_DIR}/src/csim/update_ops_matrix_dense_double.cpp
        ${qulacs_simd_path_benchmark_SOURCE_DIR}/src/csim/utility.cpp
    )
    target_include_directories(qulacs_simd_path_kernels PUBLIC
        ${qulacs_simd_path_benchmark_SOURCE_DIR}/src
        ${qulacs_simd_path_benchmark_SOURCE_DIR}/src/csim
    )
    target_compile_definitions(qulacs_simd_path_kernels PRIVATE _USE_SIMD)
    target_compile_options(qulacs_simd_path_kernels PRIVATE -march=native)
    if(SCALUQ_USE_OMP)
        find_package(OpenMP REQUIRED)
        target_link_libraries(qulacs_simd_path_kernels PUBLIC OpenMP::OpenMP_CXX)
    endif()

    add_executable(simd_path_qulacs_benchmark simd_path_qulacs_benchmark.cpp)
    target_link_libraries(simd_path_qulacs_benchmark PRIVATE
        scaluq
        qulacs_simd_path_kernels
        Eigen3::Eigen
        Kokkos::kokkos
        nlohmann_json::nlohmann_json
    )
    target_include_directories(simd_path_qulacs_benchmark PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )
endif()
