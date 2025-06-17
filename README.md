# Scaluq

**For the Japanese version of this README, see [README_ja.md](README_ja.md).**

Scaluq is a newly redeveloped Python/C++ library based on the quantum circuit simulator [Qulacs](https://github.com/qulacs/qulacs).  
It enables high-speed simulation of large-scale quantum circuits, noisy quantum circuits, and parametric quantum circuits.  
This library is released under the MIT License.

Compared to [Qulacs](https://github.com/qulacs/qulacs), the following improvements have been made:

- Implementation based on [Kokkos](https://github.com/kokkos/kokkos) allows seamless switching between execution environments (CPU/GPU) without requiring code changes.
- Improved execution speed.
- Pointers are hidden from users, making the code simpler and safer to write.
- Integration of [nanobind](https://github.com/wjakob/nanobind) enables more compact and faster Python bindings.
- Provides a faster interface for the case where the same circuit is applied to multiple quantum states.

## Build Requirements

- Ninja ≥ 1.10
- GCC ≥ 11 (≥ 13 if not using CUDA)
- CMake ≥ 3.24
- CUDA ≥ 12.6 (only when using CUDA)
- Python ≥ 3.9 (only when using Python)

Note: It may work with lower versions, but this has not been verified.

## Runtime Requirements

- CUDA ≥ 12.6 (only when using CUDA)

Note: It may work with lower versions, but this has not been verified.

## Build Options

Build options can be specified using environment variables when running `script/configure` or `pip install .`.

| Variable Name           | Default     | Description |
|------------------------|-------------|-------------|
| `CMAKE_C_COMPILER`     | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) |
| `CMAKE_CXX_COMPILER`   | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) |
| `CMAKE_BUILD_TYPE`     | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) |
| `CMAKE_INSTALL_PREFIX` | -           | See [CMake Documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html) |
| `SCALUQ_USE_OMP`       | `ON`        | Use OpenMP for parallel computation on CPU |
| `SCALUQ_USE_CUDA`      | `OFF`       | Enable parallel computation using GPU (CUDA) |
| `SCALUQ_CUDA_ARCH`     | (auto)      | Target Nvidia GPU architecture (see [Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html), e.g., `SCALUQ_CUDA_ARCH=AMPERE80`) |
| `SCALUQ_USE_TEST`      | `ON`        | Include `test/` in build targets. You can build and run tests with `ctest --test-dir build/` |
| `SCALUQ_USE_EXE`       | `ON`        | Include `exe/` in build targets. You can try running without installing by building with `ninja -C build` and running `build/exe/main` |
| `SCALUQ_FLOAT16`       | `OFF`       | Enable `f16` precision |
| `SCALUQ_FLOAT32`       | `ON`        | Enable `f32` precision |
| `SCALUQ_FLOAT64`       | `ON`        | Enable `f64` precision |
| `SCALUQ_BFLOAT16`      | `OFF`       | Enable `bf16` precision |

## Installing as a C++ Library

To install Scaluq as a static C++ library, run the following commands:

```txt
git clone https://github.com/qulacs/scaluq
cd scaluq
script/configure
sudo -E env "PATH=$PATH" ninja -C build install
```

- Required libraries such as Eigen and Kokkos will be installed together.
- You can install to a location other than `/usr/local/` by setting `CMAKE_INSTALL_PREFIX`. For example, if you want to install locally or avoid conflicts with other Kokkos builds:  
  `CMAKE_INSTALL_PREFIX=~/.local script/configure; ninja -C build install`
- `sudo` is used to install files to `/usr/local/`, but to preserve the user environment, we use `-E` and explicitly pass `PATH`.
- If you want to build the CUDA-enabled version (when NVIDIA GPU and CUDA are available), set `SCALUQ_USE_CUDA=ON`. Example:  
  `SCALUQ_USE_CUDA=ON script/configure; sudo env -E "PATH=$PATH" ninja -C build install`

When changing options and rebuilding, make sure to clear the CMake cache by running:

```txt
rm build/CMakeCache.txt
```

An example CMake configuration for a project using the installed Scaluq library is provided in [example_project/](example_project/CMakeLists.txt).

## Installing as a Python Library

Scaluq can also be used as a Python library:

```txt
pip install scaluq
```

If you want to use GPU or precisions other than `f32` and `f64`, clone the repository and install with options:

```txt
git clone https://github.com/qulacs/scaluq
cd ./scaluq
SCALUQ_USE_CUDA=ON pip install .
```

## Python Documentation

A simple documentation page is available with function descriptions and type information for the Python library version:  
https://scaluq.readthedocs.io/en/latest/index.html

## Sample Code (C++)

```cpp
#include <iostream>
#include <cstdint>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/operator/operator.hpp>
#include <scaluq/state/state_vector.hpp>

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        constexpr Precision Prec = scaluq::Precision::F64;
        constexpr ExecutionSpace Space = scaluq::ExecutionSpace::Default;
        const std::uint64_t n_qubits = 3;
        scaluq::StateVector<Prec, Default> state = scaluq::StateVector<Prec, Default>::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        scaluq::Circuit<Prec, Default> circuit(n_qubits);
        circuit.add_gate(scaluq::gate::X<Prec, Default>(0));
        circuit.add_gate(scaluq::gate::CNot<Prec, Default>(0, 1));
        circuit.add_gate(scaluq::gate::Y<Prec, Default>(1));
        circuit.add_gate(scaluq::gate::RX<Prec, Default>(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        scaluq::Operator<Prec, Default> observable(n_qubits);
        observable.add_random_operator(1, 0);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
```

By including `scaluq/all.hpp`, you can omit template arguments using `SCALUQ_OMIT_TEMPLATE`.

```cpp
#include <iostream>
#include <cstdint>

#include <scaluq/all.hpp>

namespace my_scaluq {
    SCALUQ_OMIT_TEMPLATE(scaluq::Precision::F64, scaluq::ExecutionSpace::Default)
}

using namespace my_scaluq;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        const std::uint64_t n_qubits = 3;
        StateVector state = StateVector::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        Circuit circuit(n_qubits);
        circuit.add_gate(gate::X(0));
        circuit.add_gate(gate::CNot(0, 1));
        circuit.add_gate(gate::Y(1));
        circuit.add_gate(gate::RX(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        Operator observable(n_qubits);
        observable.add_random_operator(1, 0);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
```

## Sample Code (Python)

```python
from scaluq.default.f64 import *
import math

n_qubits = 3
state = StateVector.Haar_random_state(n_qubits, 0)

circuit = Circuit(n_qubits)
circuit.add_gate(gate.X(0))
circuit.add_gate(gate.CNot(0, 1))
circuit.add_gate(gate.Y(1))
circuit.add_gate(gate.RX(1, math.pi / 2))
circuit.update_quantum_state(state)

observable = Operator(n_qubits)
observable.add_random_operator(1, 0)
value = observable.get_expectation_value(state)
print(value)
```

# Specifying Precision and Execution Space

Scaluq supports multiple floating-point precisions: `f16`, `f32`, `f64`, and `bf16`.  
By default, only `f32` and `f64` are enabled.  
While `f64` is generally recommended, lower precisions like `f32` can be up to 2–4x faster in applications such as quantum machine learning that do not require high precision.

| Precision | C++ Template Argument         | Python Submodule      | Description               |
|-----------|-------------------------------|------------------------|---------------------------|
| `f16`     | `Precision::F16`              | `f16`                  | IEEE754 binary16          |
| `f32`     | `Precision::F32`              | `f32`                  | IEEE754 binary32          |
| `f64`     | `Precision::F64`              | `f64`                  | IEEE754 binary64          |
| `bf16`    | `Precision::BF16`             | `bf16`                 | bfloat16                  |

Execution spaces determine whether computation is performed on CPU or GPU:

| Execution Space | C++ Template Argument      | Python Submodule      | Description                                |
|------------------|---------------------------|------------------------|--------------------------------------------|
| `default`        | `ExecutionSpace::Default` | `default`              | GPU if CUDA is enabled, otherwise CPU      |
| `host`           | `ExecutionSpace::Host`    | `host`                 | Always CPU                                 |

Note: You can only perform operations between objects with the same precision and execution space. For example, a gate created for 32-bit precision cannot be used with a 64-bit StateVector, even if both are CPU-based.

In C++, classes like StateVector, Circuit, Gate, and Operator accept `Precision` and `ExecutionSpace` as template arguments.

In Python, you import from submodules like `scaluq.default.f32` or `scaluq.host.f64` based on your desired configuration.

You can dynamically select the submodule using `importlib`:

```python
import importlib

prec = 'f64'
space = 'default'
scaluq_sub = importlib.import_module(f'scaluq.{space}.{prec}')
StateVector = scaluq_sub.StateVector
gate = scaluq_sub.gate

state = StateVector(3)
x = gate.X(0)
x.update_quantum_state(state)
print(state)
```
