cmake_minimum_required(VERSION 3.11)

add_library(qulacs2023)

target_sources(qulacs2023 PRIVATE
    gate/gate_one_qubit.cpp
    gate/update_ops_one_qubit.cpp
    state/state_vector.cpp
    util/random.cpp
)
target_link_libraries(qulacs2023 PRIVATE core Kokkos::kokkos)
target_include_directories(qulacs2023 PRIVATE ${PROJECT_SOURCE_DIR}/src)
