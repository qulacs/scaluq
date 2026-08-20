# SYCL on NVIDIA devcontainer

This devcontainer builds the current open-source Intel LLVM/DPC++ toolchain
with its Unified Runtime CUDA adapter.  
Verification on NVIDIA A100 AMPERE80 (sm_80).  

## Start and verify
```bash
devcontainer up --config .devcontainer/sycl-cuda/devcontainer.json
```
or
```bash
docker build -t scaluq-sycl-nvidia -f .devcontainer/sycl-cuda/Dockerfile .

docker run -it --rm --gpus 1 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e ONEAPI_DEVICE_SELECTOR="cuda:*" -e CC=/opt/dpcpp/bin/clang -e CXX=/opt/dpcpp/bin/clang++ -v $(pwd):/workspaces/scaluq -w /workspaces/scaluq scaluq-sycl-nvidia /bin/bash
```

Inside the container:  
```bash
sycl-ls
git config --global --add safe.directory /workspaces/scaluq #optional
```
`sycl-ls` should include an entry whose backend is `cuda`.  


## Modify the code

### CMakeLists.txt
```diff
### Fetch dependencies ###
# Kokkos
FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos
-    GIT_TAG 1b1383c6001f3bfe9fe309ca923c2d786600cc79 # 4.6.01
+    GIT_TAG 5.1.1
)
FetchContent_MakeAvailable(kokkos)
set_property(TARGET kokkoscore PROPERTY POSITION_INDEPENDENT_CODE ON)
```

```diff
if(SCALUQ_USE_SYCL)
    set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON CACHE BOOL "Disable PCH to avoid icpx offload bundler error")
    set(Kokkos_ENABLE_SYCL ON CACHE BOOL "Enable Kokkos SYCL backend")
+    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 --offload-arch=sm_80")
+    set(Kokkos_ENABLE_UNSUPPORTED_ARCHS ON CACHE BOOL "Enable Kokkos SYCL backend for nvidia gpu")
+    set(SCALUQ_SYCL_ARCH "AMPERE80")
    if(DEFINED SCALUQ_SYCL_ARCH)
        set(Kokkos_ARCH_${SCALUQ_SYCL_ARCH} ON)
    endif(DEFINED SCALUQ_SYCL_ARCH)
endif(SCALUQ_USE_SYCL)
```

```diff
if(SCALUQ_USE_SYCL)
    target_compile_definitions(scaluq_base PUBLIC SCALUQ_USE_SYCL)
-    target_compile_options(scaluq_base PUBLIC -fsycl)
-    target_link_options(scaluq_base PUBLIC -fsycl)
+    target_compile_options(scaluq_base PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 --offload-arch=sm_80)
+    target_link_options(scaluq_base PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 --offload-arch=sm_80)
endif()
```

> **Note**  
> `-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 --offload-arch=sm_80` and `set(SCALUQ_SYCL_ARCH "AMPERE80")`  
> are for AMPERE80. Modify arch for proper one.


### include/scaluq/kokkos.hpp
```diff
#pragma once

+#ifdef __SYCL_DEVICE_ONLY__
+extern "C" {
+    // Polyfill for missing atomic implementation in DPC++ NVIDIA backend
+    __attribute__((sycl_device))
+    __attribute__((used))
+    __attribute__((weak))
+    unsigned long _Z29__clc_atomic_compare_exchangePU3AS1mmmiii(
+        __attribute__((address_space(1))) unsigned long* ptr,
+        unsigned long cmp,
+        unsigned long val,
+        int success,
+        int failure,
+        int scope) {
+        __atomic_compare_exchange_n(ptr, &cmp, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
+        return cmp;
+    }
+}
+#endif
+#include <Kokkos_Core.hpp>

namespace scaluq {
void initialize();
void finalize();
bool is_initialized();
bool is_finalized();
void synchronize();
}  // namespace scaluq
```


## Build and Test
```txt
BUILD=build-sycl-dpcpp CC=/opt/dpcpp/bin/clang CXX=/opt/dpcpp/bin/clang++ SCALUQ_USE_SYCL=ON SCALUQ_USE_TEST=ON SCALUQ_USE_MOLD=OFF SCALUQ_USE_OMP=OFF script/configure

ninja -C build-sycl-dpcpp

./build-sycl-dpcpp/tests/scaluq_test --gtest_filter=-OperatorBatchedTest/F64DefaultSpace.Apply:OperatorBatchedTest/F32DefaultSpace.Apply
```

> **Note**  
> SCALUQ_USE_OMP=ON is not verified.  
> Scaluq (sycl-for-cuda) can use almost all functions, exclude OperatorBatched `get_applied_states`.



## Install as a Python Library
```txt
CC=/opt/dpcpp/bin/clang CXX=/opt/dpcpp/bin/clang++ SCALUQ_USE_SYCL=ON SCALUQ_USE_TEST=OFF SCALUQ_USE_EXE=OFF SCALUQ_USE_MOLD=OFF SCALUQ_USE_OMP=OFF pip install --break-system-packages -e .
```
