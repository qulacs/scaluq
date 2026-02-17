# Streams and Execution Spaces

Scaluq uses Kokkos execution spaces under the hood. In Python, each execution space instance is wrapped as {class}`ConcurrentStream <scaluq.ConcurrentStream>`. A {class}`StateVector <scaluq.default.f64.StateVector>` or {class}`Operator <scaluq.default.f64.Operator>` stores one execution space instance internally and uses it as the default queue for kernels and deep copies. The queues managed by this class represent concurrency, but they do not guarantee actual parallel execution.

## Creating Streams

Use {func}`create_default_streams <scaluq.create_default_streams>` to partition the default execution space, or {func}`create_host_streams <scaluq.create_host_streams>` to partition the host execution space.
The list values are partition weights for splitting resources across the created streams.

```py
import scaluq

s0, s1 = scaluq.create_default_streams([1.0, 1.0])
h0, h1 = scaluq.create_host_streams([1.0, 1.0])
```

## StateVector Stream Behavior

- A {class}`StateVector <scaluq.default.f64.StateVector>` owns an execution space instance.
- `copy()` uses the `StateVector`'s current stream.
- `copy(stream)` allocates and copies on the provided stream.
- `set_concurrent_stream(stream)` changes the execution space instance used by subsequent kernels; it does not reallocate or migrate data.

When you create a new object on a different stream, there is no automatic dependency between `s0` and `s1`. You should synchronize as needed based on the data hazards.

```py
import scaluq
from scaluq.default.f64 import StateVector

s0, s1 = scaluq.create_default_streams([1.0, 1.0])

psi = StateVector(s0, 3)
psi.set_computational_basis(3) # execute on stream s0

psi_same = psi.copy()     # uses s0

# Ensure s0 has finished producing data before copying from it.
scaluq.synchronize(s0)
psi_other = psi.copy(s1)  # copy on s1

psi.set_concurrent_stream(s1)
psi.normalize()
scaluq.synchronize(s1)
```

## Operator Stream Behavior

- An {class}`Operator <scaluq.default.f64.Operator>` owns an execution space instance.
- `copy()` / `get_dagger()` / `load()` / `uninitialized_operator()` use the operator’s stream by default, or the explicit `stream` overload if provided.
- Methods that take a {class}`StateVector <scaluq.default.f64.StateVector>` always use the {class}`StateVector <scaluq.default.f64.StateVector>`'s execution space, not the operator’s stream.
- Binary operators (`op1 + op2`, `op1 * op2`, etc.) run on the left-hand side operator’s stream.

```py
import scaluq
from scaluq.default.f64 import StateVector, Operator, PauliOperator

s0, s1 = scaluq.create_default_streams([1.0, 1.0])

op = Operator(s0, [PauliOperator("X 0")])
state = StateVector(s1, 1)

# Uses state’s stream (s1), not op’s stream (s0).
op.apply_to_state(state)

# If you want to run on s0, switch the stream explicitly.
state.set_concurrent_stream(s0)
op.apply_to_state(state)
```

## Stream Type Mismatch

{class}`ConcurrentStream <scaluq.ConcurrentStream>` performs a runtime type check. If you pass a stream backed by a different
execution space type, a `RuntimeError` is raised when the stream is used.

Notes:
- On CUDA builds, `scaluq.default.*` uses the device execution space, and host streams should be
  used with `scaluq.host.*` or `scaluq.host_serial.*` modules.
- On CPU-only builds, `default` and `host` are the same execution space, so host streams are valid
  for default objects.
