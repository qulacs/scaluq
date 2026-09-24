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

# Stub generation imports the target extension. Honor the same emulator used
# by C++ tests when building AVX512 wheels on a non-AVX512 host.
foreach(stub root main gate)
    set(module scaluq.scaluq_core)
    if(NOT stub STREQUAL "root")
        string(APPEND module ".default.${MAIN_PRECISION}")
    endif()
    if(stub STREQUAL "gate")
        string(APPEND module ".gate")
    endif()
    install(CODE "
        set(stub_command \"${CMAKE_CROSSCOMPILING_EMULATOR}\")
        list(APPEND stub_command \"${Python_EXECUTABLE}\" -m nanobind.stubgen
            -m \"${module}\" -o \"${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/${stub}.pyi\")
        execute_process(COMMAND \${stub_command}
            WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}\"
            COMMAND_ERROR_IS_FATAL ANY)
    ")
endforeach()
install(CODE "
    execute_process(
        COMMAND
            \"${Python_EXECUTABLE}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}/install_public_stubs.py\"
            \"${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/root.pyi\"
            \"${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/main.pyi\"
            \"${CMAKE_CURRENT_BINARY_DIR}/generated-stubs/gate.pyi\"
            \"${MAIN_PRECISION}\"
            \"\${CMAKE_INSTALL_PREFIX}\"
        COMMAND_ERROR_IS_FATAL ANY)
")
