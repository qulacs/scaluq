cmake_minimum_required(VERSION 3.11)

add_subdirectory(core)

if(USE_KOKKOS)
    add_subdirectory(kokkossim)
else()
    add_subdirectory(cpusim)
endif()
