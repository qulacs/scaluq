cmake_minimum_required(VERSION 3.11)

project(qulacs)

include(FetchContent)

### Define variables ###
if(NOT DEFINED USE_OMP)
    set(USE_OMP Yes)
endif(NOT DEFINED USE_OMP)
if(NOT DEFINED USE_CUDA)
    set(USE_CUDA No)
endif(NOT DEFINED USE_CUDA)
if(NOT DEFINED USE_KOKKOS)
    set(USE_KOKKOS No)
endif(NOT DEFINED USE_KOKKOS)
if(NOT DEFINED USE_PYTHON)
    set(USE_PYTHON No)
endif(NOT DEFINED USE_PYTHON)
if(NOT DEFINED USE_TEST)
    set(USE_TEST Yes)
endif(NOT DEFINED USE_TEST)

if(USE_CUDA AND NOT USE_KOKKOS)
    message(FATAL_ERROR "Using CUDA without Kokkos is not supported.")
endif(USE_CUDA AND NOT USE_KOKKOS)

message(STATUS "USE_PYTHON = ${USE_PYTHON}")
message(STATUS "USE_TEST = ${USE_TEST}")

### Fetch dependencies ###
# Kokkos
if(USE_KOKKOS)
    FetchContent_Declare(
        kokkos_fetch
        GIT_REPOSITORY https://github.com/kokkos/kokkos
        GIT_TAG 4.1.00
    )
    FetchContent_GetProperties(kokkos_fetch)
    if(NOT kokkos_fetch_POPULATED)
        message(STATUS "Fetch Kokkos for parallel execution")
        FetchContent_Populate(kokkos_fetch)
        add_subdirectory(${kokkos_fetch_SOURCE_DIR})
    endif(NOT kokkos_fetch_POPULATED)
endif(USE_KOKKOS)

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
if(USE_PYTHON)
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
endif(USE_PYTHON)

# Google test
if(USE_TEST)
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
endif(USE_TEST)

### Kokkos options ###
if (USE_KOKKOS)
    set(Kokkos_ENABLE_SERIAL Yes)
    if(USE_OMP)
        set(Kokkos_ENABLE_OPENMP Yes)
    endif(USE_OMP)
    if(USE_CUDA)
        set(Kokkos_ENABLE_CUDA Yes)
    endif(USE_CUDA)
endif(USE_KOKKOS)

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
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Enable pthread
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

    # Enable openmp
    if(USE_OMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        add_compile_definitions(OPENMP)
    endif()
endif()

### Add subdirectories ###
add_subdirectory(src)
if(USE_PYTHON)
    add_subdirectory(python)
endif(USE_PYTHON)
if(USE_TEST)
    add_subdirectory(tests)
endif(USE_TEST)

### Custom target ###
# test
if(USE_TEST)
    add_custom_target(
        test
        DEPENDS cpusim_test
        COMMAND cpusim_test
    )
endif(USE_TEST)

# format
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.[ch]
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
