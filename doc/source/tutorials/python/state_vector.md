# StateVector

State vector of a quantum state is expressed as {class}`StateVector <scaluq.default.f64.StateVector>`.
This holds $2^n$ complex numbers where $n$ is the number of qubits.

Indices of the qubit is numbered as $0,1,\dots,n-1$.
The state is represented by computational basis. For example `[a, b, c, d]` means $a\ket{00}+b\ket{01}+c\ket{10}+d\ket{11}$.
$i$-th bit from lower (counted from $0$) of the basis is a qubit $i$.
For example, $\ket{110}$ means that qubit $0$ is $\ket{0}$, qubit $1$ is $\ket{1}$, and qubit $2$ is $\ket{2}$.

{class}`StateVector <scaluq.default.f64.StateVector>` is constructed with single integer value `n_qubits`, which is the number of qubits.
The state is initialized to $\ket{0\dots 0}$.

```py
from scaluq.default.f64 import StateVector
state = StateVector(2)
print(state)
```
```
Qubit Count : 2
Dimension : 4
State vector : 
  00 : (1,0)
  01 : (0,0)
  10 : (0,0)
  11 : (0,0)
```

## Getting property of StateVector

{func}`n_qubits <scaluq.default.f64.StateVector>` returns the number of qubits $n$.

{func}`dim <scaluq.default.f64.StateVector>` returns the dimension of the vector $2^n$.

{func}`get_amplitudes <scaluq.default.f64.StateVector>` returns the content of StateVector as type `list[complex]`.

{func}`get_squared_norm()` returns the squared norm of the state $\braket{\phi, \phi}$ which must be equal to $1$ with normalized states.

## Initializing StateVector

In addition to the constructor, you can initialize state with some static functions.

{func}`Haar_random_state <scaluq.default.f64.StateVector>` initialize state with Haar random state. You pass seed as an optional argument. Without passing seed, the random device of system is used.

{func}`uninitialized_state <scaluq.default.f64.StateVector>` only allocate memory on the execution space without initializing. The content of the vector is undefined. It is not guaranteed to be normalized. Since allocating the vector by this function is faster than other initializing functions, you should use this if you will load other vector immediately after.

```py
from scaluq.default.f64 import StateVector
state = StateVector.Haar_random_state(2)
print("Haar random (without seed): ", state.get_amplitudes())
state = StateVector.Haar_random_state(2)
print("Haar random (without seed): ", state.get_amplitudes())
state = StateVector.Haar_random_state(2, 0)
print("Haar random (seed = 0): ", state.get_amplitudes())
state = StateVector.Haar_random_state(2, 0)
print("Haar random (seed = 0): ", state.get_amplitudes())
state = StateVector.uninitialized_state(2) # the content is undefined
```
```
Haar random (without seed):  [(-0.12377148096652951+0.027715511836463032j), (0.23343008413304153-0.6125930779810899j), (0.348536530607012+0.16314293047597564j), (-0.6344277255462337-0.059671766025997705j)]
Haar random (without seed):  [(0.20861863181020227+0.36023007097415805j), (-0.7038208261050227+0.15424679536389918j), (0.032557696049571434+0.4498796978221459j), (0.3196074684536188+0.04422729148920198j)]
Haar random (seed = 0):  [(0.24602695668676106-0.3593147366777609j), (-0.2016366688947537+0.10904346570777179j), (-0.7078115548871466+0.3479734076173536j), (0.09795534521513291-0.3551589695281517j)]
Haar random (seed = 0):  [(0.24602695668676106-0.3593147366777609j), (-0.2016366688947537+0.10904346570777179j), (-0.7078115548871466+0.3479734076173536j), (0.09795534521513291-0.3551589695281517j)]
```

## Loading StateVector

You can load the content of {class}`StateVector <scaluq.default.f64.StateVector>` by various functions.

{func}`set_zero_state <scaluq.default.f64.StateVector.set_zero_state>` is used for setting the vector to $\ket{00\dots0}=[1,0,\dots,0]$.

{func}`set_zero_norm_state <scaluq.default.f64.StateVector.set_zero_norm_state>` is used for setting the vector to $0=[0,0,\dots,0]$.

{func}`set_computational_basis <scaluq.default.f64.StateVector.set_computational_basis>` is used for setting the vector to $\ket{b}$ with input args $0\leq b \leq 2^{n}-1$.

{func}`load <scaluq.default.f64.StateVector.load>` is used for loading the other vector (with $2^n$ length) directly as amplitudes.

```py
from scaluq.default.f64 import StateVector
state = StateVector(2)
state.set_zero_state()
print("zero state:", state.get_amplitudes())
state.set_zero_norm_state()
print("zero norm state:", state.get_amplitudes())
state.load([0.5, 0.5, -0.5, 0.5])
print("loaded state:", state.get_amplitudes())
import numpy as np
state.load(np.array([1, 0, 0, 1]) / np.sqrt(2)) # You can also load numpy arrays
print("loaded numpy state:", state.get_amplitudes())
```
```
zero state: [(1+0j), 0j, 0j, 0j]
zero norm state: [0j, 0j, 0j, 0j]
loaded state: [(0.5+0j), (0.5+0j), (-0.5+0j), (0.5+0j)]
loaded numpy state: [(0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j)]
```

## Operation to StateVector

You can perform some operations to {class}`StateVector <scaluq.default.f64.StateVector>`.

{func}`add_state_vector_with_coef <scaluq.default.f64.add_state_vector_with_coef>` is used to update the vector by adding $c\ket{\psi}$, where `c` is a complex number and $\ket{\psi}$ is another state vector with same dimension.

{func}`multiply_coef <scaluq.default.f64.multiply_coef>` is used to update the vector by multiplying a complex number.

```py
import math
from scaluq.default.f64 import StateVector
phi = StateVector(2)
print("phi:", phi.get_amplitudes())
psi = StateVector.uninitialized_state(2)
psi.set_computational_basis(3)
print("psi:", psi.get_amplitudes())
phi.add_state_vector_with_coef(1j, psi)
print("phi after added psi:", phi.get_amplitudes())
phi.multiply_coef(1 / math.sqrt(2))
print("phi after multiplied coef:", phi.get_amplitudes())
```
```
phi: [(1+0j), 0j, 0j, 0j]
psi: [0j, 0j, 0j, (1+0j)]
phi after added psi: [(1+0j), 0j, 0j, 1j]
phi after multiplied coef: [(0.7071067811865475+0j), 0j, 0j, 0.7071067811865475j]
```

## Get probabilistic measures of StateVector

You can get probabilistic measures of {class}`StateVector <scaluq.default.f64.StateVector>`.

{func}`get_zero_probability <scaluq.default.f64.StateVector.get_zero_probability>` is used to get the probability of getting $0$ when the specified qubit is measured by Z-basis.

{func}`get_marginal_probability <scaluq.default.f64.StateVector.get_marginal_probability>` is used to get the marginal probability of getting specified result when some of qubits are measured simultaneously by Z-basis.
The result is specified by a list of integer with length `n`. $i$-th value of elements means as follows:
- `0`: $i$-th qubit is measured and the result is $0$.
- `1`: $i$-th qubit is measured and the result is $1$.
- {func}`StateVector.UNMEASURED <scaluq.default.f64.StateVector.UNMEASURED>`: $i$-th qubit is not measured.

{func}`get_entropy <scaluq.default.f64.StateVector.get_entropy>` is used to get the entropy of the vector, which is calculated by $\sum_i -p_i \log_2 p_i$ ($p_i$ ($0\leq i<2^n$) is $|v_i|^2$ with $v_i$ is the $i$-th amplitude of the vector).

{func}`sampling <scaluq.default.f64.StateVector.sampling>` is used to perform sampling on the vector.
With passing the number of sampling as `sampling_count`, a list of integers with length `sampling_count` is returned.

```py
import math
from scaluq.default.f64 import StateVector
state = StateVector.uninitialized_state(2)
vec = [1/2, 0, 0, math.sqrt(3)/2 * 1j]
state.load(vec)
print("zero probability of 0:", state.get_zero_probability(0))
assert abs(state.get_zero_probability(0) - (abs(vec[0])**2 + abs(vec[2])**2)) < 1e-9
print("zero probability of 1:", state.get_zero_probability(1))
assert abs(state.get_zero_probability(1) - (abs(vec[0])**2 + abs(vec[1])**2)) < 1e-9
print("marginal probability of [1, UNMEASURED]:", state.get_marginal_probability([1, StateVector.UNMEASURED]))
assert abs(state.get_marginal_probability([1, StateVector.UNMEASURED]) - (abs(vec[1])**2 + abs(vec[3])**2)) < 1e-9
```
```
zero probability of 0: 0.25
zero probability of 1: 0.25
marginal probability of [1, UNMEASURED]: 0.7499999999999999
```

## StateVectorBatched: view and copy

{class}`StateVectorBatched <scaluq.default.f64.StateVectorBatched>` stores multiple state vectors in one object.
You can extract each element as a single {class}`StateVector <scaluq.default.f64.StateVector>` in two ways:

- `view_state_vector_at(...)`: returns a view (shared memory).
- `get_state_vector_at(...)`: returns a copied state vector.

The stream overload takes {class}`ConcurrentStream <scaluq.ConcurrentStream>` as the first argument:
- `view_state_vector_at(stream, batch_id)`
- `get_state_vector_at(stream, batch_id)`

```py
import scaluq
from scaluq.default.f64 import StateVectorBatched

s0, s1 = scaluq.create_default_streams([1.0, 1.0])
states = StateVectorBatched(2, 2)
states.set_zero_norm_state()

# View: write-through to the original batched object.
view_state = states.view_state_vector_at(1)
view_state.set_amplitude_at(3, 0.25 - 0.5j)
assert states.get_state_vector_at(1).get_amplitude_at(3) == 0.25 - 0.5j

# Get: independent copy.
copied_state = states.get_state_vector_at(1)
copied_state.set_amplitude_at(3, 0.9 + 0.1j)
assert states.get_state_vector_at(1).get_amplitude_at(3) == 0.25 - 0.5j
```
