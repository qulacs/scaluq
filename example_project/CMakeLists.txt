cmake_minimum_required(VERSION 3.21)

project(example)

include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

find_package(OpenMP)
if(SCALUQ_USE_CUDA)
    find_package(CUDAToolkit)
endif()
find_package(Kokkos)
find_package(scaluq)
add_executable(main main.cpp)
target_include_directories(main PUBLIC 
    /usr/local/include/scaluq
    /usr/local/include/kokkos
    /usr/local/include/eigen3
    ${json_SOURCE_DIR}/include
)
target_compile_features(main PUBLIC cxx_std_20)
target_compile_options(main PUBLIC -fopenmp)
target_compile_definitions(main PUBLIC OPENMP)
target_link_libraries(main PUBLIC scaluq::scaluq)
