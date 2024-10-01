cmake_minimum_required(VERSION 3.21)

if(SCALUQ_USE_CUDA)
    set_source_files_properties(
        update_ops_dense_matrix.cpp
        update_ops_sparse_matrix.cpp
        PROPERTIES LANGUAGE CUDA
    )
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
endif()

target_sources(scaluq PRIVATE
    circuit/circuit.cpp
    gate/update_ops_dense_matrix.cpp
    gate/update_ops_sparse_matrix.cpp
    gate/update_ops_standard.cpp
    # gate/merge_gate.cpp
    operator/apply_pauli.cpp
    operator/pauli_operator.cpp
    operator/operator.cpp
    state/state_vector.cpp
    state/state_vector_batched.cpp
)
target_link_libraries(scaluq PRIVATE 
    Kokkos::kokkos
    Eigen3::Eigen
)
target_include_directories(scaluq PRIVATE ${PROJECT_SOURCE_DIR}/scaluq)
