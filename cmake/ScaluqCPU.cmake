# A single target selection applies to Scaluq, Kokkos and downstream consumers.
set(scaluq_simd_choices OFF AVX2 AVX512)
if(NOT SCALUQ_SIMD IN_LIST scaluq_simd_choices)
    message(FATAL_ERROR "SCALUQ_SIMD must be OFF, AVX2, or AVX512 (got '${SCALUQ_SIMD}')")
endif()
foreach(legacy SCALUQ_CPU_NATIVE SCALUQ_CPU_ARCH SCALUQ_USE_AVX512)
    if((DEFINED ${legacy} AND NOT "${${legacy}}" STREQUAL "") OR
       (DEFINED ENV{${legacy}} AND NOT "$ENV{${legacy}}" STREQUAL ""))
        message(FATAL_ERROR
            "${legacy} has been replaced by SCALUQ_SIMD=OFF|AVX2|AVX512. "
            "Remove the old option/environment variable "
            "(cmake -U '${legacy}' for an existing build directory).")
    endif()
endforeach()

set(SCALUQ_EFFECTIVE_CPU_ARCH "")
set(SCALUQ_CPU_COMPILE_OPTIONS "")
if(SCALUQ_SIMD STREQUAL "OFF" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    # Disable both explicit intrinsics and compiler-generated BMI2 instructions.
    set(SCALUQ_CPU_COMPILE_OPTIONS -mno-bmi2)
endif()
if(NOT SCALUQ_SIMD STREQUAL "OFF")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        message(FATAL_ERROR "SCALUQ_SIMD=${SCALUQ_SIMD} requires an x86-64 target")
    endif()
    if(SCALUQ_SIMD STREQUAL "AVX2")
        set(SCALUQ_EFFECTIVE_CPU_ARCH HSW)
    else()
        set(SCALUQ_EFFECTIVE_CPU_ARCH SKX)
    endif()
endif()

# Clear cached targets so switching modes also works in an existing build.
foreach(arch NATIVE HSW SKX)
    set(Kokkos_ARCH_${arch} OFF CACHE BOOL "" FORCE)
endforeach()
if(SCALUQ_EFFECTIVE_CPU_ARCH)
    set(Kokkos_ARCH_${SCALUQ_EFFECTIVE_CPU_ARCH} ON CACHE BOOL "" FORCE)
endif()
message(STATUS "SCALUQ_EFFECTIVE_CPU_ARCH = ${SCALUQ_EFFECTIVE_CPU_ARCH}")
