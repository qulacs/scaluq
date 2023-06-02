cmake_minimum_required(VERSION 3.11)

pybind11_add_module(qulacs_core SHARED EXCLUDE_FROM_ALL binding.cpp)
target_include_directories(qulacs_core PRIVATE ${PROJECT_SOURCE_DIR}/src/core)
target_include_directories(qulacs_core PRIVATE ${PROJECT_SOURCE_DIR}/src/cpusim)
target_link_libraries(qulacs_core PUBLIC core)
target_link_libraries(qulacs_core PUBLIC cpusim)
