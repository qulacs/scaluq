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

nanobind_add_stub(
    scaluq_stub
    INSTALL_TIME
    MODULE scaluq.scaluq_core
    OUTPUT scaluq/__init__.pyi
    MARKER_FILE scaluq/py.typed
    VERBOSE
)
set(PRECISIONS)
if(SCALUQ_FLOAT16)
    list(APPEND PRECISIONS f16)
endif()
if(SCALUQ_FLOAT32)
    list(APPEND PRECISIONS f32)
endif()
if(SCALUQ_FLOAT64)
    list(APPEND PRECISIONS f64)
endif()
if(SCALUQ_BFLOAT16)
    list(APPEND PRECISIONS bf16)
endif()
foreach(SPACE default host)
    set(PYTHON_MODULE_NAME scaluq.scaluq_core.${SPACE})
    if(NOT(SCALUQ_USE_CUDA) AND SPACE STREQUAL host)
        set(PYTHON_MODULE_NAME scaluq.scaluq_core.default)
    endif()
    nanobind_add_stub(
        scaluq_stub
        INSTALL_TIME
        MODULE ${PYTHON_MODULE_NAME}
        OUTPUT scaluq/${SPACE}/__init__.pyi
        MARKER_FILE scaluq/py.typed
        VERBOSE
    )
    foreach(PRECISION IN LISTS PRECISIONS)
        nanobind_add_stub(
            scaluq_stub
            INSTALL_TIME
            MODULE ${PYTHON_MODULE_NAME}.${PRECISION}
            OUTPUT scaluq/${SPACE}/${PRECISION}/__init__.pyi
            MARKER_FILE scaluq/py.typed
            VERBOSE
        )
        nanobind_add_stub(
            scaluq_stub
            INSTALL_TIME
            MODULE ${PYTHON_MODULE_NAME}.${PRECISION}.gate
            OUTPUT scaluq/${SPACE}/${PRECISION}/gate/__init__.pyi
            MARKER_FILE scaluq/py.typed
            VERBOSE
        )
    endforeach()
endforeach()
