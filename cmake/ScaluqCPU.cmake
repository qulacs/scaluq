# Preserve the existing CPU options: explicit architecture overrides native.
# Clear targets selected by Scaluq when reconfiguring an existing build tree.
foreach(arch NATIVE HSW SKX ${SCALUQ_CONFIGURED_CPU_ARCH})
    set(Kokkos_ARCH_${arch} OFF CACHE BOOL "" FORCE)
endforeach()

set(cpu_arch "")
if(SCALUQ_CPU_ARCH)
    set(cpu_arch "${SCALUQ_CPU_ARCH}")
elseif(SCALUQ_CPU_NATIVE)
    set(cpu_arch NATIVE)
endif()
if(cpu_arch)
    set(Kokkos_ARCH_${cpu_arch} ON CACHE BOOL "" FORCE)
endif()
set(SCALUQ_CONFIGURED_CPU_ARCH "${cpu_arch}" CACHE INTERNAL
    "CPU architecture last selected by Scaluq" FORCE)
