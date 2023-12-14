cmake_minimum_required(VERSION 3.21)

project(qulacs)

include(FetchContent)

### Define variables ###
if(NOT DEFINED QULACS_USE_OMP)
    set(QULACS_USE_OMP Yes)
endif(NOT DEFINED QULACS_USE_OMP)
if(NOT DEFINED QULACS_USE_CUDA)
    set(QULACS_USE_CUDA No)
endif(NOT DEFINED QULACS_USE_CUDA)
if(NOT DEFINED QULACS_USE_PYTHON)
    set(QULACS_USE_PYTHON No)
endif(NOT DEFINED QULACS_USE_PYTHON)
if(NOT DEFINED QULACS_USE_TEST)
    set(QULACS_USE_TEST Yes)
endif(NOT DEFINED QULACS_USE_TEST)

message(STATUS "QULACS_USE_PYTHON = ${QULACS_USE_PYTHON}")
message(STATUS "QULACS_USE_TEST = ${QULACS_USE_TEST}")

### Compile Warnings ###
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(WARNING_CPP "-Wall -Wdate-time -Wendif-labels -Werror=format=2 \
        -Werror=missing-declarations -Werror=return-type -Wextra \
        -Wfloat-equal -Wimplicit-fallthrough=5 -Wlogical-op \
        -Wmissing-include-dirs -Wpointer-arith -Wredundant-decls \
        -Wshadow -Wstrict-aliasing=2 -Wsuggest-attribute=noreturn -Wwrite-strings \
        -fdiagnostics-color=auto -fstrict-aliasing")
    # -Werror=undef is eliminated due to conflict with boost
elseif ((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang"))
    set(WARNING_CPP "-Wall -Wdate-time -Wendif-labels -Werror=format=2 \
        -Werror=missing-declarations -Werror=return-type -Wextra \
        -Wfloat-equal -Wimplicit-fallthrough \
        -Wmissing-include-dirs -Wpointer-arith -Wredundant-decls \
        -Wshadow -Wstrict-aliasing=2 -Wwrite-strings \
        -fdiagnostics-color=auto -fstrict-aliasing")
endif()

### Compiler options ###
if ((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang"))
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Enable pthread
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

    # Enable openmp
    if(QULACS_USE_OMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        add_compile_definitions(OPENMP)
    endif()
endif()

### Kokkos options ###
set(Kokkos_ENABLE_SERIAL Yes)
if(QULACS_USE_OMP)
    set(Kokkos_ENABLE_OPENMP Yes)
endif(QULACS_USE_OMP)
if(QULACS_USE_CUDA)
    set(Kokkos_ENABLE_CUDA Yes)
    set(Kokkos_ARCH_VOLTA70 Yes)
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

# pybind11
if(QULACS_USE_PYTHON)
    FetchContent_Declare(
        pybind11_fetch
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.10.0
    )
    FetchContent_GetProperties(pybind11_fetch)
    if(NOT pybind11_fetch_POPULATED)
        message(STATUS "Fetch pybind11 for python-binding")
        FetchContent_Populate(pybind11_fetch)
        add_subdirectory(${pybind11_fetch_SOURCE_DIR})
    endif(NOT pybind11_fetch_POPULATED)
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

### Add subdirectories ###
add_subdirectory(qulacs)
if(QULACS_USE_PYTHON)
    add_subdirectory(python)
endif(QULACS_USE_PYTHON)
if(QULACS_USE_TEST)
    add_subdirectory(tests)
endif(QULACS_USE_TEST)

### Custom target ###
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
