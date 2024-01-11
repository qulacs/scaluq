cmake_minimum_required(VERSION 3.21)

project(qulacs)

include(FetchContent)

### Define variables ###
if(NOT DEFINED QULACS_USE_OMP)
    set(QULACS_USE_OMP ON)
endif(NOT DEFINED QULACS_USE_OMP)
if(NOT DEFINED QULACS_USE_CUDA)
    set(QULACS_USE_CUDA OFF)
endif(NOT DEFINED QULACS_USE_CUDA)
if(NOT DEFINED QULACS_USE_PYTHON)
    set(QULACS_USE_PYTHON OFF)
endif(NOT DEFINED QULACS_USE_PYTHON)
if(NOT DEFINED QULACS_USE_TEST)
    set(QULACS_USE_TEST ON)
endif(NOT DEFINED QULACS_USE_TEST)
if(NOT DEFINED QULACS_USE_EXE)
    set(QULACS_USE_EXE ON)
endif(NOT DEFINED QULACS_USE_EXE)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build project as a shared library for dynamic linking")

message(STATUS "QULACS_USE_PYTHON = ${QULACS_USE_PYTHON}")
message(STATUS "QULACS_USE_TEST = ${QULACS_USE_TEST}")

### Kokkos options ###
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos Serial backend")
if(QULACS_USE_OMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend")
endif(QULACS_USE_OMP)
if(QULACS_USE_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend")
    set(Kokkos_ARCH_PASCAL61 ON CACHE BOOL "Enable Kokkos CUDA Pascal61 architecture")
    find_program(CUDA_NVCC_EXECUTABLE nvcc)
    if(CUDA_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER_WRAPPER ${CUDA_NVCC_EXECUTABLE})
        message(STATUS "Using nvcc_wrapper for CUDA compilation")
    else()
        message(SEND_ERROR "nvcc not found")
    endif()
endif(QULACS_USE_CUDA)

### Fetch dependencies ###
# Kokkos
FetchContent_Declare(
    kokkos_fetch
    GIT_REPOSITORY https://github.com/kokkos/kokkos
    GIT_TAG 4.2.00
)
FetchContent_GetProperties(kokkos_fetch)
if(NOT kokkos_fetch_POPULATED)
    message(STATUS "Fetch Kokkos for parallel execution")
    FetchContent_Populate(kokkos_fetch)
    add_subdirectory(${kokkos_fetch_SOURCE_DIR})
endif(NOT kokkos_fetch_POPULATED)

# Eigen
FetchContent_Declare(
    eigen_fetch
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 3.4.0
)
FetchContent_GetProperties(eigen_fetch)
if(NOT eigen_fetch_POPULATED)
    message(STATUS "Fetch Eigen for matrix operation")
    FetchContent_Populate(eigen_fetch)
    add_subdirectory(${eigen_fetch_SOURCE_DIR})
endif(NOT eigen_fetch_POPULATED)

# nanobind
if(QULACS_USE_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    FetchContent_Declare(
        nanobind_fetch
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG v1.8.0
    )
    FetchContent_GetProperties(nanobind_fetch)
    if(NOT nanobind_fetch_POPULATED)
        message(STATUS "Fetch nanobind for python-binding")
        FetchContent_Populate(nanobind_fetch)
        add_subdirectory(${nanobind_fetch_SOURCE_DIR})
    endif(NOT nanobind_fetch_POPULATED)
endif(QULACS_USE_PYTHON)

# Google test
if(QULACS_USE_TEST)
    FetchContent_Declare(
        googletest_fetch
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG release-1.12.1
    )
    FetchContent_GetProperties(googletest_fetch)
    if(NOT googletest_fetch_POPULATED)
        message(STATUS "Fetch googletest for C++ testing")
        FetchContent_Populate(googletest_fetch)
        add_subdirectory(${googletest_fetch_SOURCE_DIR})
    endif()
else()
    message(STATUS "Skip downloding googletest")
endif(QULACS_USE_TEST)

add_library(qulacs)

### Compiler options ###
if ((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang"))
    # Standard
    target_compile_features(qulacs PUBLIC cxx_std_20)
    
    # Warning
    target_compile_options(qulacs PUBLIC
        -Wall -Wextra -Wunused-parameter -Wshadow -pedantic
        -Wsign-compare -Wtype-limits -Wuninitialized)

    # Enable pthread
    target_compile_options(qulacs PUBLIC -pthread)

    # Enable openmp
    if(QULACS_USE_OMP)
        target_compile_options(qulacs PUBLIC -fopenmp)
        target_compile_definitions(qulacs PUBLIC OPENMP)
    endif()
    
    # Debug options
    target_compile_options(qulacs PUBLIC $<$<CONFIG:Debug>:-O0 -g -fsanitize=address,undefined>)
    target_link_options(qulacs PUBLIC $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endif()

### Add subdirectories ###
add_subdirectory(qulacs)
if(QULACS_USE_PYTHON)
    add_subdirectory(python)
endif(QULACS_USE_PYTHON)
if(QULACS_USE_TEST)
    add_subdirectory(tests)
endif(QULACS_USE_TEST)
if(QULACS_USE_EXE)
    add_subdirectory(exe)
endif(QULACS_USE_EXE)

# python
if(QULACS_USE_PYTHON)
    add_custom_target(
        python
        DEPENDS qulacs_core
    )
endif(QULACS_USE_PYTHON)

# test
if(QULACS_USE_TEST)
    add_custom_target(
        test
        DEPENDS qulacs_test
        COMMAND qulacs_test
    )
endif(QULACS_USE_TEST)

# format
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/qulacs/*.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/qulacs/*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/test/*.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/python/*.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/python/*.[ch]
    )
    add_custom_target(
        format
        COMMAND clang-format
		-style=file
        -i
        ${ALL_CXX_SOURCE_FILES}
    )
endif()
