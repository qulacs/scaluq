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
state = StateVector(2)
print(state)
```
```
```
