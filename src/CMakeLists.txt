cmake_minimum_required(VERSION 3.21)

target_sources(scaluq PRIVATE
    state/state_vector.cpp
    util/utility.cpp
    kokkos.cpp
    types.cpp
)
target_link_libraries(scaluq PRIVATE 
    Kokkos::kokkos
    Eigen3::Eigen
)
target_include_directories(scaluq PRIVATE ${PROJECT_SOURCE_DIR}/include)
