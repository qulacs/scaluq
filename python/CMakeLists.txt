cmake_minimum_required(VERSION 3.24)

message(STATUS "Building library for python...")

nanobind_add_module(scaluq_core STABLE_ABI binding.cpp)
target_link_libraries(scaluq_core PRIVATE
    scaluq
    Kokkos::kokkos
    nlohmann_json::nlohmann_json
)
target_include_directories(scaluq_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(scaluq_core PRIVATE ${eigen_SOURCE_DIR})
install(TARGETS scaluq_core LIBRARY DESTINATION scaluq)

if(SCALUQ_FLOAT64)
    set(MAIN_PRECISION f64)
elseif(SCALUQ_FLOAT32)
    set(MAIN_PRECISION f32)
elseif(SCALUQ_FLOAT16)
    set(MAIN_PRECISION f16)
elseif(SCALUQ_BFLOAT16)
    set(MAIN_PRECISION bf16)
else()
    message(FATAL_ERROR "No precision enabled for python bindings.")
endif()

nanobind_add_stub(
    scaluq_stub_root
    INSTALL_TIME
    MODULE scaluq.scaluq_core
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/root.pyi
    VERBOSE
)
nanobind_add_stub(
    scaluq_stub_main
    INSTALL_TIME
    MODULE scaluq.scaluq_core.default.${MAIN_PRECISION}
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/main.pyi
    VERBOSE
)
nanobind_add_stub(
    scaluq_stub_gate
    INSTALL_TIME
    MODULE scaluq.scaluq_core.default.${MAIN_PRECISION}.gate
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/gate.pyi
    VERBOSE
)
install(CODE "
    execute_process(
        COMMAND
            \"${Python_EXECUTABLE}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}/generate_stubs.py\"
            \"${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/root.pyi\" \"${CMAKE_INSTALL_PREFIX}/scaluq/\")
    execute_process(
        COMMAND
            \"${Python_EXECUTABLE}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}/generate_stubs.py\"
            \"${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/main.pyi\" \"${CMAKE_INSTALL_PREFIX}/scaluq/\")
    file(MAKE_DIRECTORY \"${CMAKE_INSTALL_PREFIX}/scaluq/gate/\")
    execute_process(
        COMMAND
            \"${Python_EXECUTABLE}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}/generate_stubs.py\"
            \"${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/gate.pyi\" \"${CMAKE_INSTALL_PREFIX}/scaluq/gate/\")
")
