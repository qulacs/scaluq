cmake_minimum_required(VERSION 3.11)

add_library(qulacs)

target_sources(qulacs PRIVATE
    gate/gate_one_qubit.cpp
    gate/update_ops_one_qubit.cpp
    state/state_vector.cpp
    util/random.cpp
)
target_link_libraries(qulacs PRIVATE core Kokkos::kokkos)
target_include_directories(qulacs PRIVATE ${PROJECT_SOURCE_DIR}/src)
