cmake_minimum_required(VERSION 3.24)

foreach(LIBRARY IN LISTS SCALUQ_LIBRARIES)
    target_link_libraries(${LIBRARY} PUBLIC 
        Kokkos::kokkos
        nlohmann_json::nlohmann_json
    )
    target_include_directories(${LIBRARY} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
    target_include_directories(${LIBRARY} PRIVATE ${eigen_SOURCE_DIR})
    if(${LIBRARY} STREQUAL scaluq_base)
        target_sources(${LIBRARY} PRIVATE
            base/kokkos.cpp
            base/util/utility.cpp
        )
    else()
        target_link_libraries(${LIBRARY} PRIVATE scaluq_base)
        target_sources(${LIBRARY} PRIVATE
            circuit/circuit.cpp
            gate/merge_gate.cpp
            gate/gate_matrix.cpp
            gate/gate_pauli.cpp
            gate/gate_probabilistic.cpp
            gate/gate_standard.cpp
            gate/gate.cpp
            gate/param_gate_pauli.cpp
            gate/param_gate_probabilistic.cpp
            gate/param_gate_standard.cpp
            gate/param_gate.cpp
            gate/update_ops_dense_matrix.cpp
            gate/update_ops_sparse_matrix.cpp
            gate/update_ops_standard.cpp
            state/state_vector.cpp
            state/state_vector_batched.cpp
            operator/apply_pauli.cpp
            operator/pauli_operator.cpp
            operator/operator.cpp
            util/utility.cpp
            types.cpp
        )
    endif()
endforeach()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/scaluq DESTINATION include/scaluq)
install(TARGETS scaluq ${SCALUQ_LIBRARIES} kokkos kokkoscore kokkoscontainers kokkosalgorithms kokkossimd nlohmann_json LIBDL EXPORT scaluqTargets LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
install(EXPORT scaluqTargets FILE scaluq-config.cmake DESTINATION share/cmake/scaluq/ NAMESPACE scaluq::)
