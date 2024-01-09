cmake_minimum_required(VERSION 3.11)

pybind11_add_module(qulacs_core SHARED EXCLUDE_FROM_ALL binding.cpp)
target_link_libraries(qulacs_core PRIVATE Kokkos::kokkos)
target_link_libraries(qulacs_core PUBLIC qulacs)
target_include_directories(qulacs_core PRIVATE ${PROJECT_SOURCE_DIR}/qulacs)
