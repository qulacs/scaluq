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
`add_state_vector_with_coef`, `multiply_coef`

## Get probabilistic measures of StateVector
`get_zero_property`, `get_marginal_property`, `get_entropy`, `sampling`

JSONは別でチュートリアルを作ると良さそうなのでここに書かなくてもOK。
