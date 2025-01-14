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
    set(SCALUQ_USE_TEST OFF)
endif(NOT DEFINED SCALUQ_USE_TEST)
if(NOT DEFINED SCALUQ_USE_EXE)
    set(SCALUQ_USE_EXE OFF)
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
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos
    GIT_TAG 4.2.00
)
FetchContent_MakeAvailable(kokkos)
set_property(TARGET kokkoscore PROPERTY POSITION_INDEPENDENT_CODE ON)

# Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG b396a6fbb2e173f52edb3360485dedf3389ef830 # Since Eigen has not released since 3.4.0(3 years ago), a commit hash is directly specified.
)
FetchContent_MakeAvailable(eigen)

# nlohmann_json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)
install(DIRECTORY ${json_SOURCE_DIR}/single_include/nlohmann DESTINATION include)

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
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
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

    if(SCALUQ_USE_CUDA)
        # to remove tremendous warnings of Eigen
        target_compile_options(scaluq PUBLIC
            --expt-relaxed-constexpr
            -Wno-unknown-pragmas
            -Wno-deprecated-enum-enum-conversion
            -DEIGEN_NO_CUDA
        ) 
    endif()

    # Enable pthread
    target_compile_options(scaluq PUBLIC -pthread)

    # Enable openmp
    if(SCALUQ_USE_OMP)
        target_compile_options(scaluq PUBLIC -fopenmp)
        target_compile_definitions(scaluq PUBLIC OPENMP)
    endif()
    
    # Debug options
    if (SCALUQ_USE_CUDA)
        target_compile_options(scaluq PUBLIC $<IF:$<CONFIG:Debug>,-O0 -g,-O3>)
    else()
        if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
            target_compile_options(scaluq PUBLIC $<IF:$<CONFIG:Debug>,-O0 -g,-O3>)
        else()
            target_compile_options(scaluq PUBLIC $<IF:$<CONFIG:Debug>,-O0 -g -fsanitize=address$<COMMA>undefined,-O3>)
            target_link_options(scaluq PUBLIC $<$<CONFIG:Debug>:-fsanitize=address$<COMMA>undefined>)
        endif()
    endif()
endif()

### Add subdirectories ###
add_subdirectory(src)
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
