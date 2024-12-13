cmake_minimum_required(VERSION 3.21)

project(example)

find_package(OpenMP)
if(SCALUQ_USE_CUDA)
    find_package(CUDAToolkit)
endif()
find_package(Kokkos)
find_package(scaluq)
add_executable(main main.cpp)
target_include_directories(main PUBLIC /usr/local/include/scaluq)
target_include_directories(main PUBLIC /usr/local/include/kokkos)
target_include_directories(main PUBLIC /usr/local/include/eigen3)
target_compile_features(main PUBLIC cxx_std_20)
target_compile_options(main PUBLIC -fopenmp)
target_compile_definitions(main PUBLIC OPENMP)
target_link_libraries(main PUBLIC scaluq::scaluq)

