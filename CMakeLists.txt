cmake_minimum_required(VERSION 3.21)

project(scaluq)

include(FetchContent)

### Define variables ###
if(NOT DEFINED SCALUQ_USE_OMP)
    set(SCALUQ_USE_OMP ON)
endif(NOT DEFINED SCALUQ_USE_OMP)
if(NOT DEFINED SCALUQ_USE_CUDA)
    set(SCALUQ_USE_CUDA OFF)
endif(NOT DEFINED SCALUQ_USE_CUDA)
if(NOT DEFINED SCALUQ_USE_TEST)
    set(SCALUQ_USE_TEST ON)
endif(NOT DEFINED SCALUQ_USE_TEST)
if(NOT DEFINED SCALUQ_USE_EXE)
    set(SCALUQ_USE_EXE ON)
endif(NOT DEFINED SCALUQ_USE_EXE)

message(STATUS "SKBUILD = ${SKBUILD}")
message(STATUS "SCALUQ_USE_TEST = ${SCALUQ_USE_TEST}")

### Kokkos options ###
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Kokkos Serial backend")
if(SCALUQ_USE_OMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable Kokkos OpenMP backend")
endif(SCALUQ_USE_OMP)
if(SCALUQ_USE_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Kokkos CUDA backend")
    if(DEFINED SCALUQ_CUDA_ARCH)
        set(Kokkos_ARCH_${SCALUQ_CUDA_ARCH} ON)
    endif(DEFINED SCALUQ_CUDA_ARCH)
    find_program(CUDA_NVCC_EXECUTABLE nvcc)
    if(CUDA_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER_WRAPPER ${CUDA_NVCC_EXECUTABLE})
        message(STATUS "Using nvcc_wrapper for CUDA compilation")
    else()
        message(SEND_ERROR "nvcc not found")
    endif()
endif(SCALUQ_USE_CUDA)

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
    set_property(TARGET kokkoscore PROPERTY POSITION_INDEPENDENT_CODE ON)
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

# Kokkos-Kernels
FetchContent_Declare(
    kokkoskernels_fetch
    GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels
    GIT_TAG 4.2.00
)
FetchContent_GetProperties(kokkoskernels_fetch)
if(NOT kokkoskernels_fetch_POPULATED)
    message(STATUS "Fetch kokkos-kernels for matrix operation")
    FetchContent_Populate(kokkoskernels_fetch)
    add_subdirectory(${kokkoskernels_fetch_SOURCE_DIR})
endif(NOT kokkoskernels_fetch_POPULATED)

# nanobind
if(SKBUILD)
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
    
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
endif(SKBUILD)

# Google test
if(SCALUQ_USE_TEST)
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
endif(SCALUQ_USE_TEST)

add_library(scaluq)
set_property(TARGET scaluq PROPERTY POSITION_INDEPENDENT_CODE ON)

### Compiler options ###
if ((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang"))
    # Standard
    target_compile_features(scaluq PUBLIC cxx_std_20)
    
    # Warning
    target_compile_options(scaluq PUBLIC
        -Wall
        -Wextra
        -Wunused-parameter
        # -Wshadow
        # -pedantic
        -Wsign-compare
        -Wtype-limits
        -Wuninitialized
    )

    # Enable pthread
    target_compile_options(scaluq PUBLIC -pthread)

    # Enable openmp
    if(SCALUQ_USE_OMP)
        target_compile_options(scaluq PUBLIC -fopenmp)
        target_compile_definitions(scaluq PUBLIC OPENMP)
    endif()
    
    # Debug options
    target_compile_options(scaluq PUBLIC $<IF:$<CONFIG:Debug>,-O0 -g -fsanitize=address$<COMMA>undefined,-O3>)
    target_link_options(scaluq PUBLIC $<$<CONFIG:Debug>:-fsanitize=address$<COMMA>undefined>)
endif()

### Add subdirectories ###
add_subdirectory(scaluq)
if(SKBUILD)
    add_subdirectory(python)
endif(SKBUILD)
if(SCALUQ_USE_TEST)
    add_subdirectory(tests)
endif(SCALUQ_USE_TEST)
if(SCALUQ_USE_EXE)
    add_subdirectory(exe)
endif(SCALUQ_USE_EXE)

# python
if(SKBUILD)
    add_custom_target(
        python
        DEPENDS scaluq_core
    )
endif(SKBUILD)

# test
if(SCALUQ_USE_TEST)
    add_custom_target(
        test
        DEPENDS scaluq_test
        COMMAND scaluq_test
    )
endif(SCALUQ_USE_TEST)

# format
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/scaluq/*.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/scaluq/*.[ch]
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
