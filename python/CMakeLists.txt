cmake_minimum_required(VERSION 3.21)


message(STATUS "Building library for python...")

nanobind_add_module(scaluq_core STABLE_ABI binding.cpp)
target_link_libraries(scaluq_core PRIVATE scaluq Kokkos::kokkos)
target_include_directories(scaluq_core PRIVATE ${PROJECT_SOURCE_DIR}/scaluq)
install(TARGETS scaluq_core LIBRARY DESTINATION scaluq)
