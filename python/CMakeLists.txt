cmake_minimum_required(VERSION 3.21)


message(STATUS "Building library for python...")

nanobind_add_module(qulacs_core STABLE_ABI binding.cpp)
target_link_libraries(qulacs_core PRIVATE Kokkos::kokkos)
target_link_libraries(qulacs_core PUBLIC qulacs)
target_include_directories(qulacs_core PRIVATE ${PROJECT_SOURCE_DIR}/qulacs)
install(TARGETS qulacs_core LIBRARY DESTINATION qulacs2023)
