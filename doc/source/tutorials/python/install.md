# Install Scaluq Python package

## Install from PyPI
The most simple way to install Scaluq is running below.

```python
pip install scaluq
```

The configuration of distributed package is below.

- Use OpenMP for parallel processing.
- All of the simulation is run on CPU.
    - Do not use GPU.
    - Execution Space `default` and `host` is same.
- Precision `f32` and `f64` is enabled.

## Build from source
If you want to install on the specific commit or with non-default option, you can install Scaluq from source.

The build requirements are below.

- Ninja ≥ 1.10
- GCC ≥ 11 (≥ 13 if not using CUDA)
- CMake ≥ 3.21
- CUDA ≥ 12.6 (only when using CUDA)
- Python ≥ 3.10 (only when using Python)

When using CUDA, use a host compiler version supported by your CUDA toolkit (see the CUDA Installation Guide Host Compiler Support Policy).

The build options are below.

| Variable Name           | Default     | Description |
|------------------------|-------------|-------------|
| `CMAKE_C_COMPILER`     | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) |
| `CMAKE_CXX_COMPILER`   | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) |
| `CMAKE_BUILD_TYPE`     | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) |
| `CMAKE_INSTALL_PREFIX` | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html) |
| `SCALUQ_USE_OMP`       | `ON`        | Use OpenMP for parallel computation on CPU |
| `SCALUQ_USE_CUDA`      | `OFF`       | Enable parallel computation using GPU (CUDA) |
| `SCALUQ_CPU_NATIVE`    | `ON`        | Build for native CPU architecture of builder's |
| `SCALUQ_CPU_ARCH`      | -           | Target CPU architecture (see [Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html), e.g., `SCALUQ_CPU_ARCH=SKX`) |
| `SCALUQ_CUDA_ARCH`     | (auto)      | Target Nvidia GPU architecture (see [Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html), e.g., `SCALUQ_CUDA_ARCH=AMPERE80`) |
| `SCALUQ_USE_TEST`      | `ON`        | Include `test/` in build targets. You can build and run tests with `ctest --test-dir build/` |
| `SCALUQ_USE_EXE`       | `ON`        | Include `exe/` in build targets. You can try running without installing by building with `ninja -C build` and running `build/exe/main` |
| `SCALUQ_FLOAT16`       | `OFF`       | Enable `f16` precision |
| `SCALUQ_FLOAT32`       | `ON`        | Enable `f32` precision |
| `SCALUQ_FLOAT64`       | `ON`        | Enable `f64` precision |
| `SCALUQ_BFLOAT16`      | `OFF`       | Enable `bf16` precision |

To install Scaluq, clone the git repository first and enter.

```
git clone https://github.com/qulacs/scaluq
cd scaluq
```

Then, install qulacs with passing configuration by environment variable.

```
SCALUQ_USE_CUDA=ON SCALUQ_FLOAT32=OFF pip install .
```
