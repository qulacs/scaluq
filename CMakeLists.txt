cmake_minimum_required(VERSION 3.24)

project(scaluq)

include(FetchContent)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(ScaluqUtility)

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
if(NOT DEFINED SCALUQ_FLOAT16)
    set(SCALUQ_FLOAT16 OFF)
endif(NOT DEFINED SCALUQ_FLOAT16)
if(NOT DEFINED SCALUQ_FLOAT32)
    set(SCALUQ_FLOAT32 ON)
endif(NOT DEFINED SCALUQ_FLOAT32)
if(NOT DEFINED SCALUQ_FLOAT64)
    set(SCALUQ_FLOAT64 ON)
endif(NOT DEFINED SCALUQ_FLOAT64)
if(NOT DEFINED SCALUQ_BFLOAT16)
    set(SCALUQ_BFLOAT16 OFF)
endif(NOT DEFINED SCALUQ_BFLOAT16)

message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "SCALUQ_USE_OMP = ${SCALUQ_USE_OMP}")
message(STATUS "SCALUQ_USE_CUDA = ${SCALUQ_USE_CUDA}")
message(STATUS "SCALUQ_CUDA_ARCH = ${SCALUQ_CUDA_ARCH}")
message(STATUS "SCALUQ_FLOAT16 = ${SCALUQ_FLOAT16}")
message(STATUS "SCALUQ_FLOAT32 = ${SCALUQ_FLOAT32}")
message(STATUS "SCALUQ_FLOAT64 = ${SCALUQ_FLOAT64}")
message(STATUS "SCALUQ_BFLOAT16 = ${SCALUQ_BFLOAT16}")
message(STATUS "SCALUQ_USE_TEST = ${SCALUQ_USE_TEST}")
message(STATUS "SCALUQ_USE_EXE = ${SCALUQ_USE_EXE}")

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
    GIT_TAG 1b1383c6001f3bfe9fe309ca923c2d786600cc79 # 4.6.01
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
    find_package(Python 3.9
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
    message(STATUS "Skip downloading googletest")
endif(SCALUQ_USE_TEST)

set(SCALUQ_LIBRARIES)
list(APPEND SCALUQ_LIBRARIES scaluq_base)
add_library(scaluq_base)

set(SCALUQ_PRECISION_DEFINITIONS)
set(SCALUQ_PRECISION_VALUES)
set(SCALUQ_PRECISION_NAMES)
if(SCALUQ_FLOAT16)
    list(APPEND SCALUQ_PRECISION_DEFINITIONS SCALUQ_FLOAT16)
    list(APPEND SCALUQ_PRECISION_VALUES F16)
    list(APPEND SCALUQ_PRECISION_NAMES f16)
endif()
if(SCALUQ_FLOAT32)
    list(APPEND SCALUQ_PRECISION_DEFINITIONS SCALUQ_FLOAT32)
    list(APPEND SCALUQ_PRECISION_VALUES F32)
    list(APPEND SCALUQ_PRECISION_NAMES f32)
endif()
if(SCALUQ_FLOAT64)
    list(APPEND SCALUQ_PRECISION_DEFINITIONS SCALUQ_FLOAT64)
    list(APPEND SCALUQ_PRECISION_VALUES F64)
    list(APPEND SCALUQ_PRECISION_NAMES f64)
endif()
if(SCALUQ_BFLOAT16)
    list(APPEND SCALUQ_PRECISION_DEFINITIONS SCALUQ_BFLOAT16)
    list(APPEND SCALUQ_PRECISION_VALUES BF16)
    list(APPEND SCALUQ_PRECISION_NAMES bf16)
endif()
list_max_idx(SCALUQ_PRECISION_VALUES precision_max_idx)

set(SCALUQ_SPACE_NAMES)
set(SCALUQ_SPACE_VALUES)
list(APPEND SCALUQ_SPACE_NAMES default)
list(APPEND SCALUQ_SPACE_VALUES Default)
if(SCALUQ_USE_CUDA)
    list(APPEND SCALUQ_SPACE_NAMES host)
    list(APPEND SCALUQ_SPACE_VALUES Host)
endif()
list_max_idx(SCALUQ_SPACE_VALUES space_max_idx)

foreach(space_idx RANGE 0 ${space_max_idx})
    list(GET SCALUQ_SPACE_NAMES ${space_idx} SPACE)
    list(GET SCALUQ_SPACE_VALUES ${space_idx} SPACE_VALUE)

    foreach(precision_idx RANGE 0 ${precision_max_idx})
        list(GET SCALUQ_PRECISION_DEFINITIONS ${precision_idx} DEF)
        list(GET SCALUQ_PRECISION_NAMES ${precision_idx} PRECISION)
        list(GET SCALUQ_PRECISION_VALUES ${precision_idx} PRECISION_VALUE)

        set(LIBRARY scaluq_${SPACE}_${PRECISION})
        list(APPEND SCALUQ_LIBRARIES ${LIBRARY})
        add_library(${LIBRARY})
        target_compile_definitions(${LIBRARY} PUBLIC ${DEF})
        target_compile_definitions(${LIBRARY} PRIVATE SCALUQ_INTERNAL_PREC=${PRECISION_VALUE} SCALUQ_INTERNAL_SPACE=${SPACE_VALUE})
    endforeach()
endforeach()

foreach(LIBRARY IN LISTS SCALUQ_LIBRARIES)
    set_property(TARGET ${LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(scaluq INTERFACE)
target_link_libraries(scaluq INTERFACE ${SCALUQ_LIBRARIES})

### Compiler options ###
# Standard
if(SCALUQ_USE_CUDA)
    target_compile_features(scaluq_base PUBLIC cxx_std_20)
else()
    target_compile_features(scaluq_base PUBLIC cxx_std_23)
endif()

# Warning
target_compile_options(scaluq_base PUBLIC
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
    target_compile_definitions(scaluq_base PUBLIC SCALUQ_USE_CUDA)
    target_compile_definitions(scaluq_base PUBLIC JSON_HAS_RANGES=0) # cite: https://github.com/nlohmann/json/issues/3907
    # to remove tremendous warnings of Eigen
    target_compile_options(scaluq_base PUBLIC
        -Wno-unknown-pragmas
        -Wno-deprecated-enum-enum-conversion
        --expt-relaxed-constexpr
        -Werror=cross-execution-space-call # prevent call of non __device__ function in device call
        -DEIGEN_NO_CUDA
    ) 
endif()

# Enable pthread
target_compile_options(scaluq_base PUBLIC -pthread)

# Enable openmp
if(SCALUQ_USE_OMP)
    target_compile_options(scaluq_base PUBLIC -fopenmp)
endif()

# Debug options
if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR CMAKE_CUDA_COMPILER)
target_compile_options(scaluq_base PUBLIC $<IF:$<CONFIG:Debug>,-O0 -g,-O3>)
else()
target_compile_options(scaluq_base PUBLIC $<IF:$<CONFIG:Debug>,-O0 -g -fsanitize=address$<COMMA>undefined,-O3>)
target_link_options(scaluq_base PUBLIC $<$<CONFIG:Debug>:-fsanitize=address$<COMMA>undefined>)
endif()

### Add subdirectories ###
add_subdirectory(src)
if(SKBUILD)
    add_subdirectory(python)
endif(SKBUILD)
if(SCALUQ_USE_TEST)
    enable_testing()
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
