cmake_minimum_required(VERSION 3.11)

project(qulacs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

### Define variables ###
if(NOT DEFINED USE_PYTHON)
    set(USE_PYTHON No)
endif(NOT DEFINED USE_PYTHON)
if(NOT DEFINED USE_TEST)
    set(USE_TEST Yes)
endif(NOT DEFINED USE_TEST)

message(STATUS "USE_PYTHON = ${USE_PYTHON}")

### Fetch dependencies ###
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
