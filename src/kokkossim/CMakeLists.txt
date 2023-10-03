cmake_minimum_required(VERSION 3.11)

add_library(kokkossim)

target_sources(kokkossim PRIVATE
    state_vector_kokkos.cpp
)
target_link_libraries(kokkossim PRIVATE core Kokkos::kokkos)
target_include_directories(kokkossim PRIVATE ${PROJECT_SOURCE_DIR}/src)
