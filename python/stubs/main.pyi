"""module for f64 precision"""

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Final, overload

import numpy
from numpy.typing import NDArray

import scaluq.scaluq_core
from scaluq.scaluq_core.default.f64 import (
    gate as gate,
    qasm2 as qasm2
)
import scaluq.scaluq_core.host_serial.f64


class Gate:
    """
    General class of QuantumGate.

    Notes
        Downcast to required to use gate-specific functions.
    """

    @overload
    def __init__(self, arg: SparseMatrixGate, /) -> None: ...

    @overload
    def __init__(self, arg: DenseMatrixGate, /) -> None: ...

    @overload
    def __init__(self, arg: IGate, /) -> None:
        """Upcast from `IGate`."""

    @overload
    def __init__(self, arg: GlobalPhaseGate, /) -> None:
        """Upcast from `GlobalPhaseGate`."""

    @overload
    def __init__(self, arg: XGate, /) -> None:
        """Upcast from `XGate`."""

    @overload
    def __init__(self, arg: YGate, /) -> None:
        """Upcast from `YGate`."""

    @overload
    def __init__(self, arg: ZGate, /) -> None:
        """Upcast from `ZGate`."""

    @overload
    def __init__(self, arg: HGate, /) -> None:
        """Upcast from `HGate`."""

    @overload
    def __init__(self, arg: SGate, /) -> None:
        """Upcast from `SGate`."""

    @overload
    def __init__(self, arg: SdagGate, /) -> None:
        """Upcast from `SdagGate`."""

    @overload
    def __init__(self, arg: TGate, /) -> None:
        """Upcast from `TGate`."""

    @overload
    def __init__(self, arg: TdagGate, /) -> None:
        """Upcast from `TdagGate`."""

    @overload
    def __init__(self, arg: SqrtXGate, /) -> None:
        """Upcast from `SqrtXGate`."""

    @overload
    def __init__(self, arg: SqrtXdagGate, /) -> None:
        """Upcast from `SqrtXdagGate`."""

    @overload
    def __init__(self, arg: SqrtYGate, /) -> None:
        """Upcast from `SqrtYGate`."""

    @overload
    def __init__(self, arg: SqrtYdagGate, /) -> None:
        """Upcast from `SqrtYdagGate`."""

    @overload
    def __init__(self, arg: P0Gate, /) -> None:
        """Upcast from `P0Gate`."""

    @overload
    def __init__(self, arg: P1Gate, /) -> None:
        """Upcast from `P1Gate`."""

    @overload
    def __init__(self, arg: RXGate, /) -> None:
        """Upcast from `RXGate`."""

    @overload
    def __init__(self, arg: RYGate, /) -> None:
        """Upcast from `RYGate`."""

    @overload
    def __init__(self, arg: RZGate, /) -> None:
        """Upcast from `RZGate`."""

    @overload
    def __init__(self, arg: U1Gate, /) -> None:
        """Upcast from `U1Gate`."""

    @overload
    def __init__(self, arg: U2Gate, /) -> None:
        """Upcast from `U2Gate`."""

    @overload
    def __init__(self, arg: U3Gate, /) -> None:
        """Upcast from `U3Gate`."""

    @overload
    def __init__(self, arg: SwapGate, /) -> None:
        """Upcast from `SwapGate`."""

    @overload
    def __init__(self, arg: EcrGate, /) -> None:
        """Upcast from `EcrGate`."""

    @overload
    def __init__(self, arg: MeasurementGate, /) -> None:
        """Upcast from `MeasurementGate`."""

    @overload
    def __init__(self, arg: PauliGate, /) -> None:
        """Upcast from `PauliGate`."""

    @overload
    def __init__(self, arg: PauliRotationGate, /) -> None:
        """Upcast from `PauliRotationGate`."""

    @overload
    def __init__(self, arg: scaluq.scaluq_core.host_serial.f64.SparseMatrixGate, /) -> None:
        """Upcast from `SparseMatrixGate`."""

    @overload
    def __init__(self, arg: scaluq.scaluq_core.host_serial.f64.DenseMatrixGate, /) -> None:
        """Upcast from `DenseMatrixGate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.

        Examples:
            >>> g = H(0)
            >>> print(g.gate_type())
            GateType.H
        """

    def target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits are not included.**

        Examples:
            >>> gate = CX(0, 1)
            >>> gate.target_qubit_list()
            [1]
        """

    def control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.

        Examples:
            >>> gate = CX(0, 1)
            >>> gate.control_qubit_list()
            [0]
        """

    def control_value_list(self) -> list[int]:
        """
        Get control values as `list[int]`.

        Examples:
            >>> gate = CX(0, 1)
            >>> gate.control_value_list()
            [1]
        """

    def operand_qubit_list(self) -> list[int]:
        """
        Get target and control qubits as `list[int]`.

        Examples:
            >>> gate = CX(0, 1)
            >>> gate.operand_qubit_list()
            [0, 1]
        """

    def target_qubit_mask(self) -> int:
        """
        Get target qubits as mask. **Control qubits are not included.**

        Examples:
            >>> gate = H(0, controls=[1, 2], control_values=[1, 0])
            >>> print(bin(gate.target_qubit_mask()))
            0b1
        """

    def control_qubit_mask(self) -> int:
        """
        Get control qubits as mask.

        Examples:
            >>> gate = H(0, controls=[1, 2], control_values=[1, 0])
            >>> print(bin(gate.control_qubit_mask()))
            0b110
        """

    def control_value_mask(self) -> int:
        """
        Get control values as mask.

        Examples:
            >>> gate = H(0, controls=[1, 2], control_values=[1, 0])
            >>> print(bin(gate.control_value_mask()))
            0b10
        """

    def operand_qubit_mask(self) -> int:
        """
        Get target and control qubits as mask.

        Examples:
            >>> gate = H(0, controls=[1, 2], control_values=[1, 0])
            >>> print(bin(gate.operand_qubit_mask()))
            0b111
        """

    def get_inverse(self) -> object:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.

        Examples:
            >>> s = S(0)
            >>> print(s.get_inverse())
            Gate Type: Sdag
              Target Qubits: {0}
              Control Qubits: {}
              Control Value: {}
        """

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.

        Examples:
            >>> gate = H(0, controls=[1, 2], control_values=[1, 0])
            >>> print(gate.get_matrix())
            [[ 0.70710678+0.j  0.70710678+0.j]
             [ 0.70710678+0.j -0.70710678+0.j]]
        """

    def to_string(self) -> str:
        """
        Get string representation of the gate.

        Examples:
            >>> g = H(0)
            >>> print(g.to_string())
            Gate Type: H  Target Qubits: {0}  Control Qubits: {}  Control Value: {}
        """

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.

        Examples:
            >>> g = H(0)
            >>> print(g)
            Gate Type: H  Target Qubits: {0}  Control Qubits: {}  Control Value: {}
        """

    def to_json(self) -> str:
        """
        Get JSON representation of the gate.

        Examples:
            >>> g = H(0)
            >>> print(g.to_json())
            {"control":[],"control_value":[],"target":[0],"type":"H"}
        """

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.

        Examples:
            >>> state = StateVector(2)
            >>> state.set_computational_basis(0)
            >>> H(0).update_quantum_state(state)
            >>> print(state)
            Qubit Count : 2
            Dimension : 4
            State vector :
              00 : (0.707107,0)
              01 : (0.707107,0)
              10 : (0,0)
              11 : (0,0)

            >>> states = StateVectorBatched(2, 1)
            >>> states.set_computational_basis(0)
            >>> H(0).update_quantum_state(states)
            >>> print(states)
            Qubit Count : 1
            Dimension : 2
            --------------------
            Batch_id : 0
            State vector :
              0 : (0.707107,0)
              1 : (0.707107,0)
            --------------------
            Batch_id : 1
            State vector :
              0 : (0.707107,0)
              1 : (0.707107,0)
            <BLANKLINE>
        """

    def phase(self) -> float:
        """Get `phase` property. The phase is represented as $\\gamma$."""

class ParamGate:
    """
    General class of parametric quantum gate.

    Notes:
        Downcast to required to use gate-specific functions.
    """

    @overload
    def __init__(self, arg: ParamGate, /) -> None:
        """Downcast from ParamGate."""

    @overload
    def __init__(self, arg: ParamGate, /) -> None:
        """Just copy shallowly."""

    @overload
    def __init__(self, param_gate: ParamRXGate) -> None:
        """Upcast from `ParamRXGate`."""

    @overload
    def __init__(self, param_gate: ParamRYGate) -> None:
        """Upcast from `ParamRYGate`."""

    @overload
    def __init__(self, param_gate: ParamRZGate) -> None:
        """Upcast from `ParamRZGate`."""

    @overload
    def __init__(self, param_gate: ParamPauliRotationGate) -> None:
        """Upcast from `ParamPauliRotationGate`."""

    @overload
    def __init__(self, param_gate: ParamProbabilisticGate) -> None:
        """Upcast from `ParamProbabilisticGate`."""

    def param_gate_type(self) -> scaluq.scaluq_core.ParamGateType:
        """Get parametric gate type as `ParamGateType` enum."""

    def param_coef(self) -> float:
        """Get coefficient of parameter."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits is not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control value as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits is not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control value as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """
        Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.
        """

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, param: float | Sequence[float], classical_register: ClassicalRegister | ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            param (float | Sequence[float]):
                Parameter value(s) for the gate. Should be a single float for `StateVector` and a sequence of floats (one for each batch) for `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.

        Examples:
            >>> import math
            >>> state = StateVector(2)
            >>> state.set_computational_basis(0)
            >>> ParamRX(0).update_quantum_state(state, param=math.pi / 4)
            >>> print(state)
            Qubit Count : 2
            Dimension : 4
            State vector :
              00 : (0.92388,0)
              01 : (0,-0.382683)
              10 : (0,0)
              11 : (0,0)

            >>> import math
            >>> states = StateVectorBatched(2, 1)
            >>> states.set_computational_basis(0)
            >>> ParamRX(0).update_quantum_state(states, param=[math.pi / 4, -math.pi / 4])
            >>> print(states)
            Qubit Count : 1
            Dimension : 2
            --------------------
            Batch_id : 0
            State vector :
              0 : (0.92388,0)
              1 : (0,-0.382683)
            --------------------
            Batch_id : 1
            State vector :
              0 : (0.92388,0)
              1 : (0,0.382683)
            <BLANKLINE>
        """

    def get_matrix(self, param: float) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """Get matrix representation of the gate with holding the parameter."""

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """Get string representation of the gate."""

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

class StateVector:
    """
    Vector representation of quantum state.

    Qubit index is start from 0. If the i-th value of the vector is $a_i$, the state is $\\sum_i a_i \\ket{i}$.

    Given `n_qubits: int`, construct with bases $\\ket{0\\dots 0}$ holding `n_qubits` number of qubits.

    Examples:
        >>> state1 = StateVector(2)
        >>> print(state1)
        Qubit Count : 2
        Dimension : 4
        State vector :
          00 : (1,0)
          01 : (0,0)
          10 : (0,0)
          11 : (0,0)
        <BLANKLINE>
    """

    def __init__(self, n_qubits: int) -> None:
        """
        Construct with specified number of qubits.

        Vector is initialized with computational basis $\\ket{0\\dots0}$.

        Args:
            n_qubits (int):
                number of qubits

            precision (str, optional):
                precision of the state vector

            space (str, optional):
                execution space to allocate the state vector

        Examples:
            >>> state1 = StateVector(2)
            >>> print(state1)
            Qubit Count : 2
            Dimension : 4
            State vector :
              00 : (1,0)
              01 : (0,0)
              10 : (0,0)
              11 : (0,0)
            <BLANKLINE>
        """

    @staticmethod
    def Haar_random_state(n_qubits: int, seed: int | None = None) -> StateVector:
        """
        Construct :class:`StateVector` with Haar random state.

        Args:
            n_qubits (int):
                number of qubits

            seed (int | None, optional):
                random seed

                If not specified, the value from random device is used.

            precision (str, optional):
                precision of the state vector

            space (str, optional):
                execution space to allocate the state vector

        Examples:
            >>> state = StateVector.Haar_random_state(2)
            >>> print(state.get_amplitudes()) # doctest: +SKIP
            [(-0.3188299516496241+0.6723250989136779j), (-0.253461343768224-0.22430415678425403j), (0.24998142919420457+0.33096908710840045j), (0.2991187916479724+0.2650813322096342j)]
            >>> print(StateVector.Haar_random_state(2).get_amplitudes()) # If seed is not specified, generated vector differs. # doctest: +SKIP
            [(-0.49336775961196616-0.3319437726884906j), (-0.36069529482031787+0.31413708595210815j), (-0.3654176892043237-0.10307602590749808j), (-0.18175679804035652+0.49033467421609994j)]
            >>> print(StateVector.Haar_random_state(2, 0).get_amplitudes()) # doctest: +SKIP
            [(0.030776817573663098-0.7321137912473642j), (0.5679070655936114-0.14551095055034327j), (-0.0932995615041323-0.07123201881040941j), (0.15213024630399696-0.2871374092016799j)]
            >>> print(StateVector.Haar_random_state(2, 0).get_amplitudes()) # If same seed is specified, same vector is generated. # doctest: +SKIP
            [(0.030776817573663098-0.7321137912473642j), (0.5679070655936114-0.14551095055034327j), (-0.0932995615041323-0.07123201881040941j), (0.15213024630399696-0.2871374092016799j)]
        """

    @staticmethod
    def uninitialized_state(n_qubits: int) -> StateVector:
        """
        Construct :class:`StateVector` without initializing.

        Args:
            n_qubits (int):
                number of qubits

            precision (str, optional):
                precision of the state vector

            space (str, optional):
                execution space to allocate the state vector
        """

    def set_Haar_random_state(self, seed: int | None = None) -> None:
        """
        Set Haar random state.

        Args:
            seed (int | None, optional):
                random seed

                If not specified, the value from random device is used.

        Examples:
            >>> state = StateVector(2)
            >>> state.set_Haar_random_state()
            >>> print(state.get_amplitudes()) # doctest: +SKIP
            [(-0.3188299516496241+0.6723250989136779j), (-0.253461343768224-0.22430415678425403j), (0.24998142919420457+0.33096908710840045j), (0.2991187916479724+0.2650813322096342j)]
        """

    def set_amplitude_at(self, index: int, value: complex) -> None:
        """
        Manually set amplitude at one index.

        Args:
            index (int):
                index of state vector

                This is read as binary.k-th bit of index represents k-th qubit.

            value (complex):
                amplitude value to set at index

        Examples:
            >>> state = StateVector(2)
            >>> state.get_amplitudes()
            [(1+0j), 0j, 0j, 0j]
            >>> state.set_amplitude_at(2, 3+1j)
            >>> state.get_amplitudes()
            [(1+0j), 0j, (3+1j), 0j]

        Notes:
            If you want to set amplitudes at all indices, you should use :meth:`.load`.
        """

    def get_amplitude_at(self, index: int) -> complex:
        """
        Get amplitude at one index.

        Args:
            index (int):
                index of state vector

                This is read as binary. k-th bit of index represents k-th qubit.

        Returns:
            complex:
                Amplitude at specified index

        Examples:
            >>> state = StateVector(2)
            >>> state.load([1+2j, 3+4j, 5+6j, 7+8j])
            >>> state.get_amplitude_at(0)
            (1+2j)
            >>> state.get_amplitude_at(1)
            (3+4j)
            >>> state.get_amplitude_at(2)
            (5+6j)
            >>> state.get_amplitude_at(3)
            (7+8j)

        Notes:
            If you want to get amplitudes at all indices, you should use :meth:`.get_amplitudes`.
        """

    def set_zero_state(self) -> None:
        """
        Initialize with computational basis $\\ket{00\\dots0}$.

        Examples:
            >>> state = StateVector.Haar_random_state(2)
            >>> state.get_amplitudes() # doctest: +SKIP
            [(-0.05726462181150916+0.3525270165415515j), (0.1133709060491142+0.3074930854078303j), (0.03542174692996924+0.18488950377672345j), (0.8530024105558827+0.04459332470844164j)]
            >>> state.set_zero_state()
            >>> state.get_amplitudes()
            [(1+0j), 0j, 0j, 0j]
        """

    def set_zero_norm_state(self) -> None:
        """
        Initialize with $0$ (null vector).

        Examples:
            >>> state = StateVector(2)
            >>> state.get_amplitudes()
            [(1+0j), 0j, 0j, 0j]
            >>> state.set_zero_norm_state()
            >>> state.get_amplitudes()
            [0j, 0j, 0j, 0j]
        """

    def set_computational_basis(self, basis: int) -> None:
        """
        Initialize with computational basis $\\ket{\\mathrm{basis}}$.

        Args:
            basis (int):
                basis as integer format ($0 \\leq \\mathrm{basis} \\leq 2^{\\mathrm{n\\_qubits}}-1$)

        Examples:
            >>> state = StateVector(2)
            >>> state.set_computational_basis(0) # |00>
            >>> state.get_amplitudes()
            [(1+0j), 0j, 0j, 0j]
            >>> state.set_computational_basis(1) # |01>
            >>> state.get_amplitudes()
            [0j, (1+0j), 0j, 0j]
            >>> state.set_computational_basis(2) # |10>
            >>> state.get_amplitudes()
            [0j, 0j, (1+0j), 0j]
            >>> state.set_computational_basis(3) # |11>
            >>> state.get_amplitudes()
            [0j, 0j, 0j, (1+0j)]
        """

    def get_amplitudes(self) -> list[complex]:
        """
        Get all amplitudes as `list[complex]`.

        Returns:
            list[complex]:
                amplitudes of list with len $2^{\\mathrm{n\\_qubits}}$

        Examples:
            >>> state = StateVector(2)
            >>> state.get_amplitudes()
            [(1+0j), 0j, 0j, 0j]
        """

    def n_qubits(self) -> int:
        """
        Get num of qubits.

        Returns:
            int:
                num of qubits

        Examples:
            >>> state = StateVector(2)
            >>> state.n_qubits()
            2
        """

    def dim(self) -> int:
        """
        Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).

        Returns:
            int:
                dimension of the vector

        Examples:
            >>> state = StateVector(2)
            >>> state.dim()
            4
        """

    def get_squared_norm(self) -> float:
        """
        Get squared norm of the state. $\\braket{\\psi|\\psi}$.

        Returns:
            float:
                squared norm of the state

        Examples:
            >>> v = [1+2j, 3+4j, 5+6j, 7+8j]
            >>> state = StateVector(2)
            >>> state.load(v)
            >>> state.get_squared_norm()
            204.0
            >>> sum([abs(a)**2 for a in v])
            204.0
        """

    def normalize(self) -> None:
        """
        Normalize state.

        Let $\\braket{\\psi|\\psi} = 1$ by multiplying constant.

        Examples:
            >>> v = [1+2j, 3+4j, 5+6j, 7+8j]
            >>> state = StateVector(2)
            >>> state.load(v)
            >>> norm = state.get_squared_norm()**.5
            >>> state.normalize()
            >>> state.get_amplitudes()
            [(0.07001400420140048+0.14002800840280097j), (0.21004201260420147+0.28005601680560194j), (0.3500700210070024+0.42008402520840293j), (0.4900980294098034+0.5601120336112039j)]
            >>> [a / norm for a in v]
            [(0.07001400420140048+0.14002800840280097j), (0.21004201260420147+0.28005601680560194j), (0.3500700210070024+0.42008402520840293j), (0.4900980294098034+0.5601120336112039j)]
        """

    def get_zero_probability(self, index: int) -> float:
        """
        Get the probability to observe $\\ket{0}$ at specified index.

        **State must be normalized.**

        Args:
            index (int):
                qubit index to be observed

        Returns:
            float:
                probability to observe $\\ket{0}$

        Examples:
            >>> v = [1 / 6**.5, 2j / 6**.5 * 1j, -1 / 6**.5, -2j / 6**.5]
            >>> state = StateVector(2)
            >>> state.load(v)
            >>> state.get_zero_probability(0)
            0.3333333333333334
            >>> state.get_zero_probability(1)
            0.8333333333333336
            >>> abs(v[0])**2+abs(v[2])**2
            0.3333333333333334
            >>> abs(v[0])**2+abs(v[1])**2
            0.8333333333333336
        """

    def get_marginal_probability(self, measured_values: Sequence[int]) -> float:
        """
        Get the marginal probability to observe as given.

        **State must be normalized.**

        Args:
            measured_values (list[int]):
                list with len n_qubits.

                `0`, `1` or :attr:`.UNMEASURED` is allowed for each elements. `0` or `1` shows the qubit is observed and the value is got. :attr:`.UNMEASURED` shows the the qubit is not observed.

        Returns:
            float:
                probability to observe as given

        Examples:
            >>> v = [1/4, 1/2, 0, 1/4, 1/4, 1/2, 1/4, 1/2]
            >>> state = StateVector(3)
            >>> state.load(v)
            >>> state.get_marginal_probability([0, 1, StateVector.UNMEASURED])
            0.0625
            >>> abs(v[2])**2 + abs(v[6])**2
            0.0625
        """

    def get_computational_basis_entropy(self) -> float:
        """
        Get the Shannon entropy of the Z-basis measurement distribution.

        **State must be normalized.**

        Returns:
            float:
                entropy

        Examples:
            >>> v = [1/4, 1/2, 0, 1/4, 1/4, 1/2, 1/4, 1/2]
            >>> state = StateVector(3)
            >>> state.load(v)
            >>> state.get_computational_basis_entropy()
            2.5
            >>> import math
            >>> sum(-abs(a)**2 * math.log2(abs(a)**2) for a in v if a != 0)
            2.5

        Notes:
            The result of this function differs from qulacs. This is because scaluq adopted 2 for the base of log in the definition of entropy $\\sum_i -p_i \\log p_i$ however qulacs adopted e.
        """

    def add_state_vector_with_coef(self, coef: complex, state: StateVector) -> None:
        """
        Add other state vector with multiplying the coef and make superposition.

        $\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef} \\ket{\\mathrm{state}}$.

        Args:
            coef (complex):
                coefficient to multiply to `state`

            state (:class:`StateVector`):
                state to be added

        Examples:
            >>> state1 = StateVector(1)
            >>> state1.load([1, 2])
            >>> state2 = StateVector(1)
            >>> state2.load([3, 4])
            >>> state1.add_state_vector_with_coef(2j, state2)
            >>> state1.get_amplitudes()
            [(1+6j), (2+8j)]
        """

    def multiply_coef(self, coef: complex) -> None:
        """
        Multiply coef.

        $\\ket{\\mathrm{this}}\\leftarrow\\mathrm{coef}\\ket{\\mathrm{this}}$.

        Args:
            coef (complex):
                coefficient to multiply

        Examples:
            >>> state = StateVector(1)
            >>> state.load([1, 2])
            >>> state.multiply_coef(2j)
            >>> state.get_amplitudes()
            [2j, 4j]
        """

    def sampling(self, sampling_count: int, seed: int | None = None) -> list[int]:
        """
        Sampling state vector independently and get list of computational basis

        Args:
            sampling_count (int):
                how many times to apply sampling

            seed (int | None, optional):
                random seed

                If not specified, the value from random device is used.

        Returns:
            list[int]:
                result of sampling

                list of `sampling_count` length. Each element is in $[0,2^{\\mathrm{n\\_qubits}})$

        Examples:
             >>> state = StateVector(2)
            >>> state.load([1/2, 0, -3**.5/2, 0])
            >>> state.sampling(8) # doctest: +SKIP
            [0, 2, 2, 2, 2, 0, 0, 2]
        """

    def to_string(self) -> str:
        """
        Information as `str`.

        Returns:
            str:
                information as str

        Examples:
            >>> state = StateVector(1)
            >>> state.to_string()
            'Qubit Count : 1\\nDimension : 2\\nState vector : \\n  0 : (1,0)\\n  1 : (0,0)\\n'
        """

    def load(self, other: Sequence[complex] | StateVector) -> None:
        """
        Load amplitudes from a sequence or another :class:`StateVector`.

        Args:
            other (collections.abc.Sequence[complex] | StateVector):
                amplitudes with len $2^{\\mathrm{n\\_qubits}}$ or source state vector
        """

    def copy(self) -> StateVector:
        """Return a deep copy in the same execution space."""

    def copy_to_default_space(self) -> StateVector:
        """
        Return a deep copy in the default execution space.

        Returns:
            :class:`scaluq.default.f64.StateVector`:
                Copied state vector.
        """

    def copy_to_host_space(self) -> StateVector:
        """
        Return a deep copy in the host execution space.

        Returns:
            :class:`scaluq.host.f64.StateVector`:
                Copied state vector.
        """

    @staticmethod
    def inner_product(a: StateVector, b: StateVector) -> complex:
        """
        Calculate inner product $\\braket{a | b}$.

        Args:
            a (:class:`StateVector`):
                left hand side of inner product

            b (:class:`StateVector`):
                right hand side of inner product

            precision (str, optional):
                precision of the state vector

            space (str, optional):
                execution space to allocate the state vector for calculation

        Returns:
            complex:
                inner product $\\braket{a | b}$

        Examples:
            >>> state1 = StateVector(2)
            >>> state1.load([1/2, 0, 0, 1/2]) # (|00> + |11>)/sqrt(2)
            >>> state2 = StateVector(2)
            >>> state2.load([1/2, 0, 0, -1/2]) # (|00> - |11>)/sqrt(2)
            >>> StateVector.inner_product(state1, state2)
            0j
        """

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`
        """

    UNMEASURED: Final[int] = ...
    """
    Constant used for `StateVector::get_marginal_probability` to express the the qubit is not measured.
    """

    def to_json(self) -> str:
        """
        Information as json style.

        Returns:
            str:
                information as json style

        Examples:
            >>> state = StateVector(1)
            >>> state.to_json()
            '{"amplitudes":[{"imag":0.0,"real":1.0},{"imag":0.0,"real":0.0}],"n_qubits":1}'
        """

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the state vector."""

class StateVectorBatched:
    """
    Batched vector representation of quantum state.

    Qubit index starts from 0. If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ are $b_i$, the state is $\\sum_i b_i 2^i$.

    Given `batch_size: int, n_qubits: int`, construct a batched state vector with specified batch size and qubits.

    Given `other: StateVectorBatched`, Construct a batched state vector by copying another batched state.

    Examples:
        >>> states = StateVectorBatched(3, 2)
        >>> print(states)
        Qubit Count : 2
        Dimension : 4
        --------------------
        Batch_id : 0
        State vector :
          00 : (1,0)
          01 : (0,0)
          10 : (0,0)
          11 : (0,0)
        --------------------
        Batch_id : 1
        State vector :
          00 : (1,0)
          01 : (0,0)
          10 : (0,0)
          11 : (0,0)
        --------------------
        Batch_id : 2
        State vector :
          00 : (1,0)
          01 : (0,0)
          10 : (0,0)
          11 : (0,0)
        <BLANKLINE>
    """

    def __init__(self, batch_size: int, n_qubits: int) -> None:
        """
        Construct batched state vector with specified batch size and qubits.

        Args:
            batch_size (int):
                Number of batches.

            n_qubits (int):
                Number of qubits in each state vector.

            precision (str, optional):
                precision of the state vector

            space (str, optional):
                execution space to allocate the state vector

        Examples:
            >>> states = StateVectorBatched(3, 2)
            >>> print(states)
            Qubit Count : 2
            Dimension : 4
            --------------------
            Batch_id : 0
            State vector :
              00 : (1,0)
              01 : (0,0)
              10 : (0,0)
              11 : (0,0)
            --------------------
            Batch_id : 1
            State vector :
              00 : (1,0)
              01 : (0,0)
              10 : (0,0)
              11 : (0,0)
            --------------------
            Batch_id : 2
            State vector :
              00 : (1,0)
              01 : (0,0)
              10 : (0,0)
              11 : (0,0)
            <BLANKLINE>
        """

    def n_qubits(self) -> int:
        """
        Get the number of qubits in each state vector.

        Returns:
            int:
                The number of qubits.
        """

    def dim(self) -> int:
        """
        Get the dimension of each state vector (=$2^{\\mathrm{n\\_qubits}}$).

        Returns:
            int:
                The dimension of the vector.
        """

    def batch_size(self) -> int:
        """
        Get the batch size (number of state vectors).

        Returns:
            int:
                The batch size.
        """

    def set_state_vector(self, state: StateVector) -> None:
        """
        Set all state vectors in the batch to the given state.

        Args:
            state (StateVector):
                State to set for all batches.
        """

    def set_state_vector_at(self, batch_id: int, state: StateVector) -> None:
        """
        Set the state vector at a specific batch index.

        Args:
            batch_id (int):
                Index in batch to set.

            state (StateVector):
                State to set at the specified index.
        """

    def get_state_vector_at(self, batch_id: int) -> StateVector:
        """
        Get the state vector at a specific batch index.

        Args:
            batch_id (int):
                Index in batch to get.

        Returns:
            StateVector:
                The state vector at the specified batch index.
        """

    def view_state_vector_at(self, batch_id: int) -> StateVector:
        """
        Return a view of the state vector at a specific batch index.

        Args:
            batch_id (int):
                Index in batch to view.

        Returns:
            StateVector:
                A view of the state vector at the specified batch index.
        """

    def set_zero_state(self) -> None:
        """Initialize all states to |0...0⟩."""

    def set_computational_basis(self, basis: int) -> None:
        """
        Set all states to the specified computational basis state.

        Args:
            basis (int):
                Index of the computational basis state.
        """

    def set_zero_norm_state(self) -> None:
        """Set all amplitudes to zero."""

    def set_Haar_random_state(self, set_same_state: bool, seed: int | None = None) -> None:
        """
        Initialize with Haar random states.

        Args:
            batch_size (int):
                Number of states in batch.

            n_qubits (int):
                Number of qubits per state.

            set_same_state (bool):
                Whether to set all states to the same random state.

            seed (int, optional):
                Random seed (default: random).
        """

    @staticmethod
    def Haar_random_state(batch_size: int, n_qubits: int, set_same_state: bool, seed: int | None = None) -> StateVectorBatched:
        """
        Construct :class:`StateVectorBatched` with Haar random state.

        Args:
            batch_size (int):
                Number of states in batch.

            n_qubits (int):
                Number of qubits per state.

            set_same_state (bool):
                Whether to set all states to the same random state.

            seed (int, optional):
                Random seed (default: random).

            precision (str, optional):
                precision of the state vector

            space (str, optional):
                execution space to allocate the state vector

        Returns:
            StateVectorBatched:
                New batched state vector with random states.
        """

    @staticmethod
    def uninitialized_state(batch_size: int, n_qubits: int) -> StateVectorBatched:
        """
        Construct :class:`StateVectorBatched` without initializing.

        Args:
            batch_size (int):
                Number of states in batch.

            n_qubits (int):
                number of qubits

            precision (str, optional):
                precision of the state vector

            space (str, optional):
                execution space to allocate the state vector
        """

    def get_squared_norm(self) -> list[float]:
        """
        Get squared norm for each state in the batch.

        Returns:
            list[float]:
                List of squared norms.
        """

    def normalize(self) -> None:
        """Normalize all states in the batch."""

    def get_zero_probability(self, target_qubit_index: int) -> list[float]:
        """
        Get probability of measuring |0⟩ on specified qubit for each state.

        Args:
            target_qubit_index (int):
                Index of qubit to measure.

        Returns:
            list[float]:
                Probabilities for each state in batch.
        """

    def get_marginal_probability(self, measured_values: Sequence[int]) -> list[float]:
        """
        Get marginal probabilities for specified measurement outcomes.

        Args:
            measured_values (list[int]):
                Measurement configuration.

        Returns:
            list[float]:
                Probabilities for each state in batch.
        """

    def get_computational_basis_entropy(self) -> list[float]:
        """
        Calculate the Shannon entropy of the Z-basis measurement distribution for each state.

        Returns:
            list[float]:
                Entropy values for each state.
        """

    def sampling(self, sampling_count: int, seed: int | None = None) -> list[list[int]]:
        """
        Sample from the probability distribution of each state.

        Args:
            sampling_count (int):
                Number of samples to take.

            seed (int, optional):
                Random seed (default: random).

        Returns:
            list[list[int]]:
                Samples for each state in batch.
        """

    def add_state_vector_with_coef(self, coef: complex, states: StateVectorBatched) -> None:
        """
        Add another batched state vector multiplied by a coefficient.

        Args:
            coef (complex):
                Coefficient to multiply with states.

            states (StateVectorBatched):
                States to add.
        """

    def multiply_coef(self, coef: complex) -> None:
        """
        Multiply all states by a coefficient.

        Args:
            coef (complex):
                Coefficient to multiply.
        """

    def load(self, other: Sequence[Sequence[complex]] | StateVectorBatched) -> None:
        """
        Load amplitudes from a nested sequence or another :class:`StateVectorBatched`.

        Args:
            other (list[list[complex]] | StateVectorBatched):
                amplitudes for each state or source batched state vector
        """

    def get_reduced_state(self) -> StateVector:
        """
        Get the sum of all states in the batch as a single state vector.

        Returns:
            StateVector:
                Reduced state vector.
        """

    def get_amplitudes(self) -> list[list[complex]]:
        """
        Get amplitudes of all states in batch.

        Returns:
            list[list[complex]]:
                Amplitudes for each state.
        """

    def copy(self) -> StateVectorBatched:
        """
        Create a deep copy of this batched state vector.

        Returns:
            StateVectorBatched:
                New copy of the states.
        """

    def copy_to_default_space(self) -> StateVectorBatched:
        """
        Return a deep copy in the default execution space.

        Returns:
            :class:`scaluq.default.f64.StateVectorBatched`:
                Copied batched state vector.
        """

    def copy_to_host_space(self) -> StateVectorBatched:
        """
        Return a deep copy in the host execution space.

        Returns:
            :class:`scaluq.host.f64.StateVectorBatched`:
                Copied batched state vector.
        """

    def to_string(self) -> str:
        """
        Get string representation of the batched states.

        Returns:
            str:
                String representation of states.

        Examples:
            >>> states = StateVectorBatched(2, 3)
            >>> print(states.to_string())
            Qubit Count : 3
            Dimension : 8
            --------------------
            Batch_id : 0
            State vector :
              000 : (1,0)
              001 : (0,0)
              010 : (0,0)
              011 : (0,0)
              100 : (0,0)
              101 : (0,0)
              110 : (0,0)
              111 : (0,0)
            --------------------
            Batch_id : 1
            State vector :
              000 : (1,0)
              001 : (0,0)
              010 : (0,0)
              011 : (0,0)
              100 : (0,0)
              101 : (0,0)
              110 : (0,0)
              111 : (0,0)
            <BLANKLINE>
        """

    def __str__(self) -> str:
        """
        Get string representation of the batched states.

        Returns:
            str:
                String representation of states.
        """

    def to_json(self) -> str:
        """
        Convert states to JSON string.

        Returns:
            str:
                JSON representation of states.

        Examples:
            >>> states = StateVectorBatched(2, 3)
            >>> print(states.to_json())
            {"amplitudes":[[{"imag":0.0,"real":1.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}],[{"imag":0.0,"real":1.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}]],"batch_size":2,"n_qubits":3}
        """

    def load_json(self, json_str: str) -> None:
        """
        Load states from JSON string.

        Args:
            json_str (str):
                JSON string to load from.
        """

class DensityMatrix:
    """
    DensityMatrix representation of quantum state.

    Qubit index is start from 0. If the (i,j)-th value of the matrix is $a_{ij}$, the state is $\\sum_{i,j} a_{ij} \\ket{i}\\bra{j}$.
    """

    @overload
    def __init__(self, n_qubits: int) -> None:
        """
        Construct with specified number of qubits.

        Matrix is initialized with computational basis $\\ket{0\\dots0}\\bra{0\\dots0}$.

        Args:
            n_qubits (int):
                number of qubits

        Examples:
            >>> state = DensityMatrix(2)
            >>> print(state)
            Qubit Count : 2
            Dimension : 4
            Is Hermitian : true
            Density Matrix :
              (00, 00) : (1,0)
              (00, 01) : (0,0)
              (00, 10) : (0,0)
              (00, 11) : (0,0)
              (01, 00) : (0,0)
              (01, 01) : (0,0)
              (01, 10) : (0,0)
              (01, 11) : (0,0)
              (10, 00) : (0,0)
              (10, 01) : (0,0)
              (10, 10) : (0,0)
              (10, 11) : (0,0)
              (11, 00) : (0,0)
              (11, 01) : (0,0)
              (11, 10) : (0,0)
              (11, 11) : (0,0)
            <BLANKLINE>
        """

    @overload
    def __init__(self, state_vector: StateVector) -> None:
        """
        Construct from state vector. The density matrix is initialized as $\\ket{\\psi}\\bra{\\psi}$ where $\\ket{\\psi}$ is the input state vector.

        Args:
            state_vector (StateVector):
                state vector to be converted to density matrix

        Examples:
            >>> sv = StateVector(2)
            >>> sv.set_computational_basis(1)
            >>> dm = DensityMatrix(sv)
            >>> print(dm.get_matrix())
            [[0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    def n_qubits(self) -> int:
        """
        Get the number of qubits.

        Returns:
            int:
                number of qubits

        Examples:
            >>> state = DensityMatrix(3)
            >>> state.n_qubits()
            3
        """

    def dim(self) -> int:
        """
        Get the dimension of the density matrix. ($=2^\\mathrm{n\\_qubits}$).

        Returns:
            int:
                dimension of the density matrix

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.dim()
            4
        """

    def is_hermitian(self) -> bool:
        """
        Check if the density matrix is guaranteed to be Hermitian.

        Returns:
            bool:
                True if the density matrix is Hermitian, False if it may not be Hermitian

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.is_hermitian()
            True
        """

    def force_hermitian(self) -> None:
        """
        Force the density matrix to be treated as Hermitian. This may enable certain optimizations but should only be used if you are sure the matrix is actually Hermitian.

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.multiply_coef(1j)
            >>> state.multiply_coef(-1j)
            >>> state.is_hermitian()
            False
            >>> state.force_hermitian()
            >>> state.is_hermitian()
            True
        """

    def get_coherence_at(self, row_index: int, col_index: int) -> complex:
        """
        Get coherence at specified position.

        This is a very slow operation. Use get_matrix() instead if possible.

        Args:
            row_index (int):
                row index

            col_index (int):
                column index

        Returns:
            complex:
                coherence at the specified position

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.get_coherence_at(0, 0)
            (1+0j)
            >>> state.get_coherence_at(0, 1)
            0j
        """

    def set_coherence_at(self, row_index: int, col_index: int, value: complex) -> None:
        """
        Set coherence at specified position.

        This is a very slow operation. Use load() instead if possible.

        is_hermitian is set to False unless diagonal element and real value is passed in.

        Args:
            row_index (int):
                row index

            col_index (int):
                column index

            value (complex):
                value to set at the specified position

        Examples:
            >>> state = DensityMatrix(2)
            >>> print(state.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
            >>> state.set_coherence_at(0, 1, 0.5+0.5j)
            >>> print(state.get_matrix())
            [[1. +0.j  0.5+0.5j 0. +0.j  0. +0.j ]
             [0. +0.j  0. +0.j  0. +0.j  0. +0.j ]
             [0. +0.j  0. +0.j  0. +0.j  0. +0.j ]
             [0. +0.j  0. +0.j  0. +0.j  0. +0.j ]]
        """

    def set_coherence_pair_at(self, row_index: int, col_index: int, value: complex) -> None:
        """
        Set coherence at specified position and its conjugate position to maintain Hermitian property.

        This is a very slow operation. Use load() instead if possible.

        Args:
            row_index (int):
                row index

            col_index (int):
                column index

            value (complex):
                value to set at the specified position

        Examples:
            >>> state = DensityMatrix(2)
            >>> print(state.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
            >>> state.set_coherence_pair_at(0, 1, 0.5+0.5j)
            >>> print(state.get_matrix())
            [[1. +0.j  0.5+0.5j 0. +0.j  0. +0.j ]
             [0.5-0.5j 0. +0.j  0. +0.j  0. +0.j ]
             [0. +0.j  0. +0.j  0. +0.j  0. +0.j ]
             [0. +0.j  0. +0.j  0. +0.j  0. +0.j ]]
        """

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get the density matrix as ndarray.

        Returns:
            Annotated[numpy.typing.NDArray[numpy.complex128], dict(shape=(None, None), order=’C’)]:
                density matrix as 2D array

        Examples:
            >>> state = DensityMatrix(2)
            >>> print(state.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    def copy(self) -> DensityMatrix:
        """
        Get a copy of the density matrix.

        Returns:
            DensityMatrix:
                a copy of the density matrix

        Examples:
            >>> state1 = DensityMatrix(2)
            >>> state2 = state1.copy()
            >>> print(state1.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
            >>> print(state2.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    def copy_to_default_space(self) -> DensityMatrix:
        """
        Get a copy of the density matrix in the default execution space.

        Returns:
            :class:`scaluq.default.f64.DensityMatrix`:
                a copy of the density matrix in the default execution space

        Examples:
            >>> state_host = DensityMatrix(2)
            >>> state_default = state_host.copy_to_default_space()
            >>> print(state_default.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    def copy_to_host_space(self) -> DensityMatrix:
        """
        Get a copy of the density matrix in the host execution space.

        Returns:
            :class:`scaluq.host.f64.DensityMatrix`:
                a copy of the density matrix in the host execution space

        Examples:
            >>> state_default = DensityMatrix(2)
            >>> state_host = state_default.copy_to_host_space()
            >>> print(state_host.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    @overload
    def load(self, matrix: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')], is_hermitian: bool = False) -> None:
        """
        Load the density matrix from a 2D array.

        Args:
            matrix (Annotated[numpy.typing.NDArray[numpy.complex128], dict(shape=(None, None), order=’C’)]):
                2D array representing the density matrix

            is_hermitian (bool):
                Whether the input matrix is guaranteed to be Hermitian.

        Examples:
            >>> import numpy as np
            >>> state = DensityMatrix(1)
            >>> matrix = np.array([[0.5+0.5j, 0], [0, 0.5-0.5j]])
            >>> state.load(matrix, is_hermitian=True)
            >>> print(state.get_matrix())
            [[0.5+0.5j 0. +0.j ]
             [0. +0.j  0.5-0.5j]]
        """

    @overload
    def load(self, other: DensityMatrix) -> None:
        """
        Load the density matrix from another DensityMatrix.

        Args:
            other (DensityMatrix):
                DensityMatrix to load from

        Examples:
            >>> state1 = DensityMatrix.Haar_random_state(1)
            >>> print(state1.get_matrix()) # doctest: +SKIP
            [[ 0.35920411+8.30722924e-18j -0.29205502-3.80631554e-01j]
             [-0.29205502+3.80631554e-01j  0.64079589+5.48595618e-18j]]
            >>> state2 = DensityMatrix(1)
            >>> state2.load(state1)
            >>> print(state2.get_matrix()) # doctest: +SKIP
            [[ 0.35920411+8.30722924e-18j -0.29205502-3.80631554e-01j]
             [-0.29205502+3.80631554e-01j  0.64079589+5.48595618e-18j]]
        """

    @overload
    def load(self, state_vector: StateVector) -> None:
        """
        Load the density matrix from a StateVector. The density matrix is set to $\\ket{\\psi}\\bra{\\psi}$ where $\\ket{\\psi}$ is the input state vector.

        Args:
            state_vector (StateVector):
                StateVector to load from

        Examples:
            >>> sv = StateVector(2)
            >>> sv.set_computational_basis(1)
            >>> dm = DensityMatrix(2)
            >>> dm.load(sv)
            >>> print(dm.get_matrix())
            [[0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    @staticmethod
    def uninitialized_state(n_qubits: int) -> DensityMatrix:
        """
        Create an uninitialized density matrix with specified number of qubits.

        Args:
            n_qubits (int):
                number of qubits

        Returns:
            DensityMatrix:
                an uninitialized density matrix
        """

    @staticmethod
    def Haar_random_state(n_qubits: int, seed: int | None = None) -> DensityMatrix:
        """
        Create a density matrix representing a Haar random state.

        Args:
            n_qubits (int):
                number of qubits

            seed (int | None, optional):
                random seed

                If not specified, the value from random device is used.

        Returns:
            DensityMatrix:
                a density matrix representing a Haar random state

        Examples:
            >>> state = DensityMatrix.Haar_random_state(1)
            >>> print(state.get_matrix()) # doctest: +SKIP
            [[ 0.35920411+8.30722924e-18j -0.29205502-3.80631554e-01j]
             [-0.29205502+3.80631554e-01j  0.64079589+5.48595618e-18j]]
            >>> state1 = DensityMatrix.Haar_random_state(1, seed=42)
            >>> state2 = DensityMatrix.Haar_random_state(1, seed=42)
            >>> print(state1.get_matrix()) # doctest: +SKIP
            [[ 0.7662367 -2.34701458e-17j -0.35360255-2.32558070e-01j]
             [-0.35360255+2.32558070e-01j  0.2337633 -7.47914693e-19j]]
            >>> print(state2.get_matrix()) # doctest: +SKIP
            [[ 0.7662367 -2.34701458e-17j -0.35360255-2.32558070e-01j]
             [-0.35360255+2.32558070e-01j  0.2337633 -7.47914693e-19j]]
        """

    def set_zero_state(self) -> None:
        """
        Set the density matrix to the zero state $\\ket{0\\dots0}\\bra{0\\dots0}$.

        Examples:
            >>> state = DensityMatrix.uninitialized_state(2)
            >>> state.set_zero_state()
            >>> print(state.get_matrix())
            [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    def set_zero_norm_state(self) -> None:
        """
        Set the density matrix to the 0 (zero matrix).

        Examples:
            >>> state = DensityMatrix.uninitialized_state(2)
            >>> state.set_zero_norm_state()
            >>> print(state.get_matrix())
            [[0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    def set_computational_basis(self, basis_index: int) -> None:
        """
        Set the density matrix to a computational basis state $\\ket{b}\\bra{b}$ where $b$ is the binary representation of the input basis index.

        Args:
            basis_index (int):
                index of the computational basis state to set (0-based)

        Examples:
            >>> state = DensityMatrix.uninitialized_state(2)
            >>> state.set_computational_basis(2) # sets state to |10><10|
            >>> print(state.get_matrix())
            [[0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
             [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
        """

    def set_Haar_random_state(self, seed: int | None = None) -> None:
        """
        Set the density matrix to represent a Haar random state.

        Args:
            seed (int | None, optional):
                random seed

                If not specified, the value from random device is used.

        Examples:
            >>> state = DensityMatrix(1)
            >>> state.set_Haar_random_state(seed=42)
            >>> print(state.get_matrix()) # doctest: +SKIP
            [[ 0.7662367 -2.34701458e-17j -0.35360255-2.32558070e-01j]
             [-0.35360255+2.32558070e-01j  0.2337633 -7.47914693e-19j]]
        """

    def get_trace(self) -> complex:
        """
        Calculate the trace of the density matrix.

        Returns:
            complex:
                trace of the density matrix

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.get_trace()
            (1+0j)
        """

    def get_partial_trace(self, traced_out_qubits: Sequence[int]) -> DensityMatrix:
        """
        Calculate the partial trace of the density matrix by tracing out the specified qubits.

        Args:
            traced_out_qubits (collections.abc.Sequence[int]):
                indices of qubits to be traced out

        Returns:
            DensityMatrix:
                the resulting density matrix after partial trace

        Examples:
            >>> state = DensityMatrix.Haar_random_state(2)
            >>> print(state.get_matrix()) # doctest: +SKIP
            [[ 0.51491987-2.74660016e-17j -0.23762498-1.56281696e-01j
               0.10531599+3.49755601e-01j -0.13503799+1.31271109e-01j]
             [-0.23762498+1.56281696e-01j  0.15709162+1.84452504e-20j
              -0.15475438-1.29440927e-01j  0.02247559-1.01563879e-01j]
             [ 0.10531599-3.49755601e-01j -0.15475438+1.29440927e-01j
               0.25910912-1.49078030e-18j  0.06154578+1.18572309e-01j]
             [-0.13503799-1.31271109e-01j  0.02247559+1.01563879e-01j
               0.06154578-1.18572309e-01j  0.06887938-1.44012992e-19j]]
            >>> reduced_state = state.get_partial_trace([1])
            >>> print(reduced_state.get_matrix()) # doctest: +SKIP
            [[ 0.774029  -2.89567819e-17j -0.17607919-3.77093869e-02j]
             [-0.17607919+3.77093869e-02j  0.225971  -1.25567742e-19j]]
        """

    def normalize(self) -> None:
        """
        Normalize the density matrix so that its trace becomes 1.

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.multiply_coef(2)
            >>> state.normalize()
            >>> print(state.get_trace())
            (1+0j)
        """

    def get_purity(self) -> float:
        """
        Calculate the purity of the quantum state represented by the density matrix. Purity is defined as $\\mathrm{Tr}(\\rho^2)$ where $\\rho$ is the density matrix.

        Returns:
            float:
                purity of the quantum state

        Examples:
            >>> state1 = DensityMatrix.Haar_random_state(2, 0)
            >>> print(state1.get_purity()) # doctest: +SKIP
            1.0000000000000002
            >>> state2 = DensityMatrix.Haar_random_state(2, 1)
            >>> state1.multiply_coef(0.5)
            >>> state1.add_density_matrix_with_coef(0.5, state2)
            >>> print(state1.get_purity()) # doctest: +SKIP
            0.6238584782007198

        Notes:
            The matrix must be hermitian and normalized
        """

    def get_zero_probability(self, target_qubit_index: int) -> float:
        """
        Calculate the probability of measuring 0 on the target qubit.

        Args:
            target_qubit_index (int):
                index of the target qubit

        Returns:
            float:
                probability of measuring 0 on the target qubit

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.set_computational_basis(2) # sets state to |10><10|
            >>> state.get_zero_probability(0)
            1.0
            >>> state.get_zero_probability(1)
            0.0

        Notes:
            The matrix must be hermitian and normalized
        """

    def get_marginal_probability(self, measured_values: Sequence[int]) -> float:
        """
        Get the marginal probability to observe as given.

        Args:
            measured_values (list[int]):
                list with len n_qubits.

                `0`, `1` or :attr:`.UNMEASURED` is allowed for each elements. `0` or `1` shows the qubit is observed and the value is got. :attr:`.UNMEASURED` shows the the qubit is not observed.

        Returns:
            float:
                probability to observe as given

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.set_computational_basis(2) # sets state to |10><10|
            >>> state.get_marginal_probability([0, 1])
            1.0
            >>> state.get_marginal_probability([0, DensityMatrix.UNMEASURED])
            1.0
            >>> state.get_marginal_probability([DensityMatrix.UNMEASURED, 1])
            1.0
            >>> state.get_marginal_probability([1, DensityMatrix.UNMEASURED])
            0.0

        Notes:
            The matrix must be hermitian and normalized
        """

    def sampling(self, sampling_count: int, seed: int | None = None) -> list[int]:
        """
        Sampling density matrix independently and get list of computational basis

        Args:
            sampling_count (int):
                number of samples to draw

            seed (int | None, optional):
                random seed

                If not specified, the value from random device is used.

        Returns:
            List[int]:
                list of sampled computational basis represented as integers

        Examples:
            >>> import numpy as np
            >>> state = DensityMatrix.uninitialized_state(1)
            >>> state.load(np.array([[0.5, -0.5j], [0.5j, 0.5]]), True)
            >>> print(state.sampling(10, seed=42)) # doctest: +SKIP
            [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]

        Notes:
            The matrix must be hermitian and normalized
        """

    def get_computational_basis_entropy(self) -> float:
        """
        Calculate the computational basis entropy of the quantum state represented by the density matrix. Computational basis entropy is defined as $-\\sum_i p_i \\log_2 p_i$ where $p_i$ is the probability of measuring the computational basis state $\\ket{i}$, which can be calculated as the (i,i)-th element of the density matrix.

        Returns:
            float:
                computational basis entropy of the quantum state

        Examples:
            >>> state = DensityMatrix(2)
            >>> state.set_computational_basis(2) # sets state to |10><10|
            >>> state.get_computational_basis_entropy()
            0.0
            >>> import numpy as np
            >>> state = DensityMatrix.uninitialized_state(1)
            >>> state.load(np.array([[0.5, -0.5j], [0.5j, 0.5]]), True)
            >>> state.get_computational_basis_entropy()
            1.0

        Notes:
            The matrix must be hermitian and normalized
        """

    def add_density_matrix_with_coef(self, coef: complex, other: DensityMatrix) -> None:
        """
        Add another density matrix to this density matrix with a coefficient. This performs the operation $\\rho \\leftarrow \\rho + c \\sigma$ where $\\rho$ is this density matrix, $c$ is the coefficient and $\\sigma$ is the other density matrix.

        Args:
            coef (complex):
                coefficient to multiply the other density matrix

            other (DensityMatrix):
                the other density matrix to be added

        Examples:
            >>> state1 = DensityMatrix.Haar_random_state(1, 1)
            >>> state2 = DensityMatrix.Haar_random_state(1, 2)
            >>> print(state1.get_matrix()) # doctest: +SKIP
            [[ 0.51823805+2.35020398e-18j -0.20611927-4.55172737e-01j]
             [-0.20611927+4.55172737e-01j  0.48176195+2.36023838e-19j]]
            >>> print(state2.get_matrix()) # doctest: +SKIP
            [[0.20777367+3.60776697e-18j 0.36552629-1.76051983e-01j]
             [0.36552629+1.76051983e-01j 0.79222633-1.66856281e-17j]]
            >>> state1.add_density_matrix_with_coef(0.5, state2)
            >>> print(state1.get_matrix()) # doctest: +SKIP
            [[ 0.62212489+4.15408747e-18j -0.02335612-5.43198728e-01j]
             [-0.02335612+5.43198728e-01j  0.87787511-8.10679023e-18j]]
        """

    def multiply_coef(self, coef: complex) -> None:
        """
        Multiply this density matrix by a coefficient. This performs the operation $\\rho \\leftarrow c \\rho$ where $\\rho$ is this density matrix and $c$ is the coefficient.

        Args:
            coef (complex):
                coefficient to multiply the density matrix

        Examples:
            >>> state = DensityMatrix.Haar_random_state(1, 1)
            >>> print(state.get_matrix()) # doctest: +SKIP
            [[ 0.51823805+2.35020398e-18j -0.20611927-4.55172737e-01j]
             [-0.20611927+4.55172737e-01j  0.48176195+2.36023838e-19j]]
            >>> state.multiply_coef(2)
            >>> print(state.get_matrix()) # doctest: +SKIP
            [[ 1.0364761 +4.70040797e-18j -0.41223854-9.10345474e-01j]
             [-0.41223854+9.10345474e-01j  0.9635239 +4.72047677e-19j]]
        """

    def to_string(self) -> str:
        """Convert the density matrix to a string representation."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`
        """

    def to_json(self) -> str:
        """
        Information as json style.

        Returns:
            str:
                information as json style

        Examples:
            >>> state = DensityMatrix(2)
            >>> print(state.to_json())
            {"is_hermitian":true,"matrix":[[{"imag":0.0,"real":1.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}],[{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}],[{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}],[{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}]],"n_qubits":2}
        """

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the density matrix."""

    UNMEASURED: Final[int] = ...
    """
    Constant used for `DensityMatrix::get_marginal_probability` to express the the qubit is not measured.
    """

class SparseMatrixGate:
    """
    Specific class of sparse matrix gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class DenseMatrixGate:
    """
    Specific class of dense matrix gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

def merge_gate(arg0: Gate, arg1: Gate, /) -> tuple[Gate, float]:
    """Merge two gates. return value is (merged gate, global phase)."""

class Operator:
    """
    General quantum operator class.

    Given `qubit_count: int`, Initialize operator with specified number of qubits.

    Examples:
        >>> terms = [PauliOperator("X 0 Y 2"), PauliOperator("Z 1 X 3", 2j)]
        >>> op = Operator(terms)
        >>> pauli = PauliOperator("X 1", -1j)
        >>> op *= pauli
        >>> print(op.to_json())
        {"terms":[{"coef":{"imag":-1.0,"real":0.0},"pauli_string":"X 0 X 1 Y 2"},{"coef":{"imag":2.0,"real":0.0},"pauli_string":"Y 1 X 3"}]}
    """

    @overload
    def __init__(self, n_terms: int) -> None:
        """Initialize operator with specified number of terms."""

    @overload
    def __init__(self, terms: Sequence[PauliOperator]) -> None:
        """Initialize operator with given list of terms."""

    class GroundState:
        """Structure to hold the ground state information."""

        @property
        def eigenvalue(self) -> complex: ...

        @property
        def state(self) -> StateVector: ...

    def is_hermitian(self) -> bool:
        """Check if the operator is Hermitian."""

    def force_hermitian(self) -> None:
        """Operator is_hermitian is true forcibly."""

    def load(self, terms: Sequence[PauliOperator]) -> None:
        """Load the operator with a list of Pauli operators."""

    @staticmethod
    def uninitialized_operator(n_terms: int) -> Operator:
        """Create an uninitialized operator with a specified number of terms."""

    def n_terms(self) -> int:
        """Get the number of terms in the operator."""

    def get_terms(self) -> list[PauliOperator]:
        """Get the list of Pauli terms that make up the operator."""

    @overload
    def to_string(self) -> str: ...

    @overload
    def to_string(self) -> str:
        """Get string representation of the operator."""

    def optimize(self) -> None:
        """Optimize the operator by combining like terms."""

    def get_dagger(self) -> Operator:
        """Get the adjoint (Hermitian conjugate) of the operator."""

    def apply_to_state(self, state: StateVector) -> None:
        """Apply the operator to a state vector."""

    @overload
    def get_expectation_value(self, state: StateVector) -> complex:
        """
        Get the expectation value of the operator with respect to a state vector.
        """

    @overload
    def get_expectation_value(self, states: StateVectorBatched) -> list[complex]:
        """
        Get the expectation values of the operator for a batch of state vectors.
        """

    def get_expectation_values(self, state: StateVector) -> list[complex]:
        """
        Get the expectation values of the pauli operators with respect to a state vector.
        """

    @overload
    def get_transition_amplitude(self, source: StateVector, target: StateVector) -> complex:
        """
        Get the transition amplitude of the operator between two state vectors.
        """

    @overload
    def get_transition_amplitude(self, states_source: StateVectorBatched, states_target: StateVectorBatched) -> list[complex]:
        """
        Get the transition amplitudes of the operator for a batch of state vectors.
        """

    def get_full_matrix(self, arg: int, /) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the Operator. Tensor product is applied from n_qubits-1 to 0.
        """

    def solve_ground_state_by_power_method(self, initial_state: StateVector, iter_count: int, mu: complex | None = None) -> Operator.GroundState:
        """
        Solve for the ground state using the power method.

        Args:
            initial_state (:class:`StateVector`):
                Initial state vector for the iteration. Passing Haar_random_state is often nice.

            iter_count (int):
                Number of iterations to perform.

            mu (complex, optional):
                Shift value to accelerate convergence. The value must be larger than largest eigenvalue. If not provided, a default value is calculated internally.

        Returns:
            :class:`Operator.GroundState`:
                The ground state information including eigenvalue and ground state vector.

        Examples:
            >>> terms = [PauliOperator("", -3.8505), PauliOperator("X 1", -0.2288), PauliOperator("Z 1", -1.0466), PauliOperator("X 0", -0.2288), PauliOperator("X 0 X 1", 0.2613), PauliOperator("X 0 Z 1", 0.2288), PauliOperator("Z 0", -1.0466), PauliOperator("Z 0 X 1", 0.2288), PauliOperator("Z 0 Z 1", 0.2356)]
            >>> op = Operator(terms)
            >>> op *= .5
            >>> ground_state = op.solve_ground_state_by_power_method(StateVector.Haar_random_state(2, seed=0), 200)
            >>> ground_state.eigenvalue.real # doctest: +ELLIPSIS
            -2.862620764...
            >>> print(ground_state.state.get_amplitudes()) # doctest: +SKIP
            [(0.593378420...-0.801949191...j), (-0.00937288681...+0.0126674290...j), (-0.00937288706...+0.0126674292...j), (-0.0389261878...+0.0526086285...j)]
        """

    def solve_ground_state_by_arnoldi_method(self, initial_state: StateVector, iter_count: int, mu: complex | None = None) -> Operator.GroundState:
        """
        Solve for the ground state using the Arnoldi method.

        Args:
            initial_state (:class:`StateVector`):
                Initial state vector for the iteration. Passing Haar_random_state is often nice.

            iter_count (int):
                Number of iterations to perform. Too large value causes error on calculating eigenvalue of Hessenberg matrix.

            mu (complex, optional):
                Shift value to accelerate convergence. The value must be larger than largest eigenvalue. If not provided, a default value is calculated internally.

        Returns:
            :class:`Operator.GroundState`:
                The ground state information including eigenvalue and ground state vector.

        Examples:
            >>> terms = [PauliOperator("", -3.8505), PauliOperator("X 1", -0.2288), PauliOperator("Z 1", -1.0466), PauliOperator("X 0", -0.2288), PauliOperator("X 0 X 1", 0.2613), PauliOperator("X 0 Z 1", 0.2288), PauliOperator("Z 0", -1.0466), PauliOperator("Z 0 X 1", 0.2288), PauliOperator("Z 0 Z 1", 0.2356)]
            >>> op = Operator(terms)
            >>> op *= .5
            >>> ground_state = op.solve_ground_state_by_arnoldi_method(StateVector.Haar_random_state(2, seed=0), 40)
            >>> ground_state.eigenvalue.real # doctest: +ELLIPSIS
            -2.862620764...
            >>> ground_state.state.get_amplitudes() # doctest: +SKIP
            [(0.593378420...-0.801949191...j), (-0.00937288693...+0.0126674291...j), (-0.00937288693...+0.0126674291...j), (-0.0389261878...+0.0526086285...j)]

        Notes:
            This works even if it is not Hermitian. An eigenvalue with the smallest real part is returned as the ground state.
        """

    def calculate_default_mu(self) -> complex:
        """
        Calculate a default shift value `mu` for ground state solvers.

        Returns:
            complex:
                Calculated shift value `mu`.
        """

    def copy(self) -> Operator:
        """Return a deep copy in the same execution space."""

    def copy_to_default_space(self) -> Operator:
        """Return a deep copy in the default execution space."""

    def copy_to_host_space(self) -> Operator:
        """Return a deep copy in the host execution space."""

    @overload
    def __imul__(self, arg: complex, /) -> Operator: ...

    @overload
    def __imul__(self, arg: PauliOperator, /) -> Operator: ...

    @overload
    def __imul__(self, arg: Operator, /) -> Operator: ...

    @overload
    def __mul__(self, arg: complex, /) -> Operator: ...

    @overload
    def __mul__(self, arg: Operator, /) -> Operator: ...

    @overload
    def __mul__(self, arg: PauliOperator, /) -> Operator: ...

    def __pos__(self) -> Operator: ...

    def __neg__(self) -> Operator: ...

    @overload
    def __add__(self, arg: Operator, /) -> Operator: ...

    @overload
    def __add__(self, arg: PauliOperator, /) -> Operator: ...

    @overload
    def __sub__(self, arg: Operator, /) -> Operator: ...

    @overload
    def __sub__(self, arg: PauliOperator, /) -> Operator: ...

    @overload
    def __iadd__(self, arg: Operator, /) -> Operator: ...

    @overload
    def __iadd__(self, arg: PauliOperator, /) -> Operator: ...

    @overload
    def __isub__(self, arg: Operator, /) -> Operator: ...

    @overload
    def __isub__(self, arg: PauliOperator, /) -> Operator: ...

    def to_json(self) -> str:
        """Information as json style."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the operator."""

    def __str__(self) -> str:
        """Get string representation of the operator."""

class OperatorBatched:
    """General quantum operator class for batched operators."""

    @overload
    def __init__(self) -> None:
        """Default constructor."""

    @overload
    def __init__(self, arg: Sequence[Sequence[PauliOperator]], /) -> None:
        """Constructor from a vector of Pauli operators."""

    @overload
    def __init__(self, arg: Sequence[Operator], /) -> None:
        """Constructor from a vector of Operators."""

    def copy(self) -> OperatorBatched:
        """Return a copy."""

    def load(self, terms: Sequence[Sequence[PauliOperator]] | Sequence[Operator]) -> None:
        """
        Load a vector of Pauli operators or a vector of Operators.

        Args:
            terms (list[list[PauliOperator]] | list[Operator]):
                terms to load
        """

    @overload
    def to_string(self) -> str:
        """Return a string representation of the operator."""

    @overload
    def to_string(self) -> str:
        """Get string representation of the OperatorBatched."""

    def batch_size(self) -> int:
        """Return the batch size of the operator."""

    def get_dagger(self) -> OperatorBatched:
        """Return the Hermitian conjugate of the operator."""

    def get_applied_states(self, state_vector: StateVector, batch_size: int = 1) -> StateVectorBatched:
        """
        Apply the batched operator to a state vector.

        Args:
            state_vector (A state vector to be applied.):
            batch_size (In the update process, the batch size to be processed simultaneously.):
        """

    @overload
    def get_expectation_value(self, state_vector: StateVector) -> list[complex]:
        """
        Return a vector of expectation values for each operator.

        Args:
            state_vector (A state vector to compute expectation values.):
        """

    @overload
    def get_expectation_value(self, states: StateVectorBatched) -> list[complex]:
        """
        Return a vector of expectation values for each operator.

        Args:
            states (State Vector Batched to compute expectation values.):
        """

    @overload
    def get_transition_amplitude(self, state_vector_bra: StateVector, state_vector_ket: StateVector) -> list[complex]:
        """
        Return a vector of transition amplitudes for each operator.

        Args:
            state_vector_bra (A bra state vector.):
            state_vector_ket (A ket state vector.):
        """

    @overload
    def get_transition_amplitude(self, states_bra: StateVectorBatched, states_ket: StateVectorBatched) -> list[complex]:
        """
        Return a vector of transition amplitudes for each operator.

        Args:
            states_bra (Batched bra state vector.):
            states_ket (Batched ket state vector.):
        """

    def get_operator_at(self, index: int) -> Operator:
        """
        Return the operator at the specified index.

        Args:
            index (Index of the operator.):
        """

    def view_operator_at(self, index: int) -> Operator:
        """
        Return a view of the operator at the specified index.

        Args:
            index (Index of the operator.):
        """

    def get_operators(self) -> list[Operator]:
        """Return a vector of all operators."""

    def copy_to_default_space(self) -> OperatorBatched:
        """Return a deep copy in the default execution space."""

    def copy_to_host_space(self) -> OperatorBatched:
        """Return a deep copy in the host execution space."""

    @overload
    def __mul__(self, arg: Sequence[complex], /) -> OperatorBatched: ...

    @overload
    def __mul__(self, arg: OperatorBatched, /) -> OperatorBatched: ...

    @overload
    def __mul__(self, arg: Sequence[PauliOperator], /) -> OperatorBatched: ...

    @overload
    def __imul__(self, arg: Sequence[complex], /) -> OperatorBatched: ...

    @overload
    def __imul__(self, arg: Sequence[PauliOperator], /) -> OperatorBatched: ...

    @overload
    def __add__(self, arg: OperatorBatched, /) -> OperatorBatched: ...

    @overload
    def __add__(self, arg: Sequence[PauliOperator], /) -> OperatorBatched: ...

    @overload
    def __sub__(self, arg: OperatorBatched, /) -> OperatorBatched: ...

    @overload
    def __sub__(self, arg: Sequence[PauliOperator], /) -> OperatorBatched: ...

    def to_json(self) -> str:
        """Information as json style."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the OperatorBatched."""

    def __str__(self) -> str:
        """Get string representation of the OperatorBatched."""

class IGate:
    """
    Specific class of I gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class GlobalPhaseGate:
    """
    Specific class of gate, which rotate global phase, represented as $e^{i\\gamma}I$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class XGate:
    """
    Specific class of Pauli-X gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class YGate:
    """
    Specific class of Pauli-Y gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class ZGate:
    """
    Specific class of Pauli-Z gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class HGate:
    """
    Specific class of Hadamard gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class SGate:
    """
    Specific class of S gate, represented as $\\begin{bmatrix} 1 & 0 \\\\ 0 & i \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class SdagGate:
    """
    Specific class of inverse of S gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class TGate:
    """
    Specific class of T gate, represented as $\\begin{bmatrix} 1 & 0 \\\\ 0 &e^{i \\pi/4} \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class TdagGate:
    """
    Specific class of inverse of T gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class SqrtXGate:
    """
    Specific class of sqrt(X) gate, represented as $\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1+i & 1-i\\\\ 1-i & 1+i \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class SqrtXdagGate:
    """
    Specific class of inverse of sqrt(X) gate, represented as $\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1-i & 1+i\\\\ 1+i & 1-i \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class SqrtYGate:
    """
    Specific class of sqrt(Y) gate, represented as $\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1+i & -1-i \\\\ 1+i & 1+i \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class SqrtYdagGate:
    """
    Specific class of inverse of sqrt(Y) gate, represented as $\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1-i & 1-i\\\\ -1+i & 1-i \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class P0Gate:
    """
    Specific class of projection gate to $\\ket{0}$.

    Notes:
        This gate is not unitary.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class P1Gate:
    """
    Specific class of projection gate to $\\ket{1}$.

    Notes:
        This gate is not unitary.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class RXGate:
    """
    Specific class of X rotation gate, represented as $e^{-i\\frac{\\theta}{2}X}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def angle(self) -> float:
        """Get `angle` property."""

class RYGate:
    """
    Specific class of Y rotation gate, represented as $e^{-i\\frac{\\theta}{2}Y}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def angle(self) -> float:
        """Get `angle` property."""

class RZGate:
    """
    Specific class of Z rotation gate, represented as $e^{-i\\frac{\\theta}{2}Z}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def angle(self) -> float:
        """Get `angle` property."""

class U1Gate:
    """
    Specific class of IBMQ's U1 Gate, which is a rotation about Z-axis, represented as $\\begin{bmatrix} 1 & 0 \\\\ 0 & e^{i\\lambda} \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class U2Gate:
    """
    Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, represented as $\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\ e^{i\\phi} & e^{i(\\phi+\\lambda)} \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def phi(self) -> float:
        """Get `phi` property."""

class U3Gate:
    """
    Specific class of IBMQ's U3 Gate, which is a rotation about 3 axis, represented as $\\begin{bmatrix} \\cos \\frac{\\theta}{2} & -e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\ e^{i\\phi}\\sin\\frac{\\theta}{2} & e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2} \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def theta(self) -> float:
        """Get `theta` property."""

    def phi(self) -> float:
        """Get `phi` property."""

class SwapGate:
    """
    Specific class of two-qubit swap gate.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

class EcrGate:
    """
    Specific class of two-qubit ecr gate.represented as $\\frac{1}{\\sqrt{2}}\\begin{bmatrix} 0 & 1 & 0 & i \\\\ 1 & 0 & -i & 0 \\\\ 0 & i & 0 & 1 \\\\ -i & 0 & 1 & 0 \\end{bmatrix}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def physical_control(self) -> list[int]:
        """Get `physical_control_indices` property."""

    def physical_target(self) -> list[int]:
        """Get `physical_target_indices` property."""

class MeasurementGate:
    """
    Specific class of computational-basis measurement gate.

    Notes:
        This gate is not unitary and requires a classical register when applied.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def classical_bit_index(self) -> int:
        """Get `classical_bit_index` property."""

    def reset(self) -> bool:
        """Return whether this measurement resets the target qubit to |0>."""

class PauliGate:
    """
    Specific class of multi-qubit pauli gate, which applies single-qubit Pauli gate to each of qubit.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def pauli(self) -> PauliOperator: ...

    def pauli_id_list(self) -> list[int]: ...

class PauliRotationGate:
    """
    Specific class of multi-qubit pauli-rotation gate, represented as $e^{-i\\frac{\\theta}{2}P}$.

    Notes
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: Gate, /) -> None:
        """Downcast from `Gate`."""

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """Get gate type as `GateType` enum."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits are not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control values as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits are not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control values as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """Generate inverse gate as `Gate` type. If not exists, return None."""

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the gate. Note: The matrix is constructed by reordering target qubits in ascending order of their indices. The qubit with the smaller index is treated as the first target, and the larger as the second.
        """

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """
        Information as `str`.

        Same as :meth:`.to_string()`.
        """

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def pauli(self) -> PauliOperator: ...

    def pauli_id_list(self) -> list[int]: ...

    def angle(self) -> float: ...

class ParamRXGate:
    """
    Specific class of parametric X rotation gate, represented as $e^{-i\\frac{\\mathrm{\\theta}}{2}X}$. `theta` is given as `param * param_coef`.

    Notes:
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    @overload
    def __init__(self, arg: ParamRXGate, /) -> None:
        """Downcast from ParamGate."""

    @overload
    def __init__(self, arg: ParamGate, /) -> None: ...

    def param_gate_type(self) -> scaluq.scaluq_core.ParamGateType:
        """Get parametric gate type as `ParamGateType` enum."""

    def param_coef(self) -> float:
        """Get coefficient of parameter."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits is not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control value as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits is not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control value as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """
        Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.
        """

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, param: float | Sequence[float], classical_register: ClassicalRegister | ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            param (float | Sequence[float]):
                Parameter value(s) for the gate. Should be a single float for `StateVector` and a sequence of floats (one for each batch) for `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def get_matrix(self, param: float) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """Get matrix representation of the gate with holding the parameter."""

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """Get string representation of the gate."""

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

class ParamRYGate:
    """
    Specific class of parametric Y rotation gate, represented as $e^{-i\\frac{\\mathrm{\\theta}}{2}Y}$. `theta` is given as `param * param_coef`.

    Notes:
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    @overload
    def __init__(self, arg: ParamRYGate, /) -> None:
        """Downcast from ParamGate."""

    @overload
    def __init__(self, arg: ParamGate, /) -> None: ...

    def param_gate_type(self) -> scaluq.scaluq_core.ParamGateType:
        """Get parametric gate type as `ParamGateType` enum."""

    def param_coef(self) -> float:
        """Get coefficient of parameter."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits is not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control value as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits is not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control value as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """
        Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.
        """

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, param: float | Sequence[float], classical_register: ClassicalRegister | ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            param (float | Sequence[float]):
                Parameter value(s) for the gate. Should be a single float for `StateVector` and a sequence of floats (one for each batch) for `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def get_matrix(self, param: float) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """Get matrix representation of the gate with holding the parameter."""

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """Get string representation of the gate."""

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

class ParamRZGate:
    """
    Specific class of parametric Z rotation gate, represented as $e^{-i\\frac{\\mathrm{\\theta}}{2}Z}$. `theta` is given as `param * param_coef`.

    Notes:
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    @overload
    def __init__(self, arg: ParamRZGate, /) -> None:
        """Downcast from ParamGate."""

    @overload
    def __init__(self, arg: ParamGate, /) -> None: ...

    def param_gate_type(self) -> scaluq.scaluq_core.ParamGateType:
        """Get parametric gate type as `ParamGateType` enum."""

    def param_coef(self) -> float:
        """Get coefficient of parameter."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits is not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control value as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits is not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control value as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """
        Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.
        """

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, param: float | Sequence[float], classical_register: ClassicalRegister | ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            param (float | Sequence[float]):
                Parameter value(s) for the gate. Should be a single float for `StateVector` and a sequence of floats (one for each batch) for `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def get_matrix(self, param: float) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """Get matrix representation of the gate with holding the parameter."""

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """Get string representation of the gate."""

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

class ParamPauliRotationGate:
    """
    Parametric multi-qubit pauli-rotation gate, represented as $e^{-i\\frac{\\theta}{2}P}$. `theta` is given as `param * param_coef`.

    Notes:
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    @overload
    def __init__(self, arg: ParamPauliRotationGate, /) -> None:
        """Downcast from ParamGate."""

    @overload
    def __init__(self, arg: ParamGate, /) -> None: ...

    def param_gate_type(self) -> scaluq.scaluq_core.ParamGateType:
        """Get parametric gate type as `ParamGateType` enum."""

    def param_coef(self) -> float:
        """Get coefficient of parameter."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits is not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control value as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits is not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control value as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """
        Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.
        """

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, param: float | Sequence[float], classical_register: ClassicalRegister | ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            param (float | Sequence[float]):
                Parameter value(s) for the gate. Should be a single float for `StateVector` and a sequence of floats (one for each batch) for `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def get_matrix(self, param: float) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """Get matrix representation of the gate with holding the parameter."""

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """Get string representation of the gate."""

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def pauli(self) -> PauliOperator: ...

    def pauli_id_list(self) -> list[int]: ...

class ParamProbabilisticGate:
    """
    Specific class of parametric probabilistic gate. The gate to apply is picked from a certain distribution.

    Notes:
        Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    @overload
    def __init__(self, arg: ParamProbabilisticGate, /) -> None:
        """Downcast from ParamGate."""

    @overload
    def __init__(self, arg: ParamGate, /) -> None: ...

    def param_gate_type(self) -> scaluq.scaluq_core.ParamGateType:
        """Get parametric gate type as `ParamGateType` enum."""

    def param_coef(self) -> float:
        """Get coefficient of parameter."""

    def target_qubit_list(self) -> list[int]:
        """Get target qubits as `list[int]`. **Control qubits is not included.**"""

    def control_qubit_list(self) -> list[int]:
        """Get control qubits as `list[int]`."""

    def control_value_list(self) -> list[int]:
        """Get control value as `list[int]`."""

    def operand_qubit_list(self) -> list[int]:
        """Get target and control qubits as `list[int]`."""

    def target_qubit_mask(self) -> int:
        """Get target qubits as mask. **Control qubits is not included.**"""

    def control_qubit_mask(self) -> int:
        """Get control qubits as mask."""

    def control_value_mask(self) -> int:
        """Get control value as mask."""

    def operand_qubit_mask(self) -> int:
        """Get target and control qubits as mask."""

    def get_inverse(self) -> object:
        """
        Generate inverse parametric-gate as `ParamGate` type. If not exists, return None.
        """

    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, param: float | Sequence[float], classical_register: ClassicalRegister | ClassicalRegisterBatched | None = None, seed: int | None = None) -> None:
        """
        Apply gate to `state`. `state` in args is directly updated.

        Optionally pass a matching classical register and seed.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated. Can be `StateVector` or `StateVectorBatched`.

            param (float | Sequence[float]):
                Parameter value(s) for the gate. Should be a single float for `StateVector` and a sequence of floats (one for each batch) for `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used in the gate. If not provided, a new register with appropriate size is created and used internally.

            seed (int | None):
                Seed for random number generator. If not provided, a random seed is generated using `std::random_device`.
        """

    def get_matrix(self, param: float) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """Get matrix representation of the gate with holding the parameter."""

    def to_string(self) -> str:
        """Get string representation of the gate."""

    def __str__(self) -> str:
        """Get string representation of the gate."""

    def to_json(self) -> str:
        """Get JSON representation of the gate."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the gate."""

    def gate_list(self) -> list[Gate | ParamGate]: ...

    def distribution(self) -> list[float]: ...

class Circuit:
    """
    Quantum circuit representation.

    Examples:
        >>> circuit = Circuit()
        >>> print(circuit.to_json())
        {"gate_list":[]}
    """

    def __init__(self) -> None:
        """Initialize empty circuit."""

    def gate_list(self) -> list[Gate | tuple[ParamGate, str]]:
        """
        Get property of `gate_list`.

        Returns:
            list:
                List of gates.

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_gate(gate.X(0))
            >>> len(circuit.gate_list())
            1
        """

    def n_gates(self) -> int:
        """
        Get property of `n_gates`.

        Returns:
            int:
                Property of `n_gates`.

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_gate(gate.X(0))
            >>> circuit.add_gate(gate.CX(0, 1))
            >>> print(circuit.n_gates())
            2
        """

    def key_set(self) -> set[str]:
        """
        Get set of keys of parameters.

        Returns:
            set[str]:
                Set of keys.

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_param_gate(gate.ParamRX(0, 0.0), "theta")
            >>> circuit.add_param_gate(gate.ParamRY(1, 2.0), "phi")
            >>> print(sorted(circuit.key_set()))
            ['phi', 'theta']
        """

    def get_gate_at(self, index: int) -> Gate | tuple[ParamGate, str]:
        """
        Get reference of i-th gate.

        Args:
            index (int):
                Index of gate

        Returns:
            Gate:
                Gate at i-th index

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_gate(gate.X(0))
            >>> print(circuit.get_gate_at(0))
            Gate Type: X
              Target Qubits: {0}
              Control Qubits: {}
              Control Value: {}
        """

    def get_param_key_at(self, index: int) -> str | None:
        """
        Get parameter key of i-th gate. If it is not parametric, return None.

        Args:
            index (int):
                Index of gate

        Returns:
            str | None:
                Parameter key

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_param_gate(gate.ParamRX(0, 0.0), "theta")
            >>> circuit.add_gate(gate.X(1))
            >>> circuit.get_param_key_at(0)
            'theta'
            >>> circuit.get_param_key_at(1) is None
            True
        """

    def get_classical_condition_at(self, index: int) -> Callable[[scaluq.scaluq_core.ClassicalRegister], bool] | None:
        """
        Get classical register condition.

        Args:
            index (int):
                index of classical register

        Returns:
            bool:
                classical bit
        """

    def calculate_depth(self) -> int:
        """
        Get depth of circuit.

        Returns:
            int:
                Depth of circuit

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_gate(gate.X(0))
            >>> circuit.add_gate(gate.Y(1))
            >>> circuit.add_gate(gate.CX(0, 1))
            >>> print(circuit.calculate_depth())
            2
        """

    def add_gate(self, gate: Gate) -> None:
        """
        Add gate to circuit.

        Args:
            gate (Gate):
                Gate to add

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_gate(gate.X(0))
        """

    @overload
    def add_conditional_gate(self, gate: Gate, condition: Callable[[scaluq.scaluq_core.ClassicalRegister], bool]) -> None:
        """Add gate with a user-defined classical condition."""

    @overload
    def add_conditional_gate(self, gate: Gate, classical_bit_index: int, expected_value: bool) -> None:
        """Add gate with a condition on a classical bit."""

    def add_param_gate(self, param_gate: ParamGate, param_key: str) -> None:
        """
        Add parametric gate with specifying key. Given param_gate is copied.

        Args:
            param_gate (ParamGate):
                Parametric gate to add

            param_key (str):
                Parameter key

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_param_gate(gate.ParamRX(0, 0.0), "theta")
        """

    @overload
    def add_conditional_param_gate(self, param_gate: ParamGate, param_key: str, condition: Callable[[scaluq.scaluq_core.ClassicalRegister], bool]) -> None:
        """Add parametric gate with a user-defined classical condition."""

    @overload
    def add_conditional_param_gate(self, param_gate: ParamGate, param_key: str, classical_bit_index: int, expected_value: bool) -> None:
        """Add parametric gate with a condition on a classical bit."""

    def add_circuit(self, other: Circuit) -> None:
        """
        Add all gates in specified circuit. Given gates are copied.

        Args:
            other (Circuit):
                Circuit to add

        Examples:
            >>> circuit = Circuit()
            >>> circuit.add_gate(gate.X(0))
            >>> circuit2 = Circuit()
            >>> circuit2.add_gate(gate.Y(1))
            >>> circuit.add_circuit(circuit2)
            >>> circuit.n_gates()
            2
        """

    def copy(self) -> Circuit:
        """
        Copy circuit. Returns a new circuit instance with all gates copied by reference.
        """

    def get_inverse(self) -> Circuit:
        """Get inverse of circuit. All the gates are newly created."""

    def to_json(self) -> str:
        """Information as json style."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the circuit."""

    @overload
    def update_quantum_state(self, state: StateVector | StateVectorBatched, *, params: Mapping[str, float] | Mapping[str, Sequence[float]] | None = None, classical_register: scaluq.scaluq_core.ClassicalRegister | scaluq.scaluq_core.ClassicalRegisterBatched | None = None, seed: int | None = None, **kwargs) -> None:
        """
        Apply circuit to `state`. `state` in args is directly updated.

        Parameters can be passed as a dict or keyword arguments. For batched states, parameter values must be sequences with length equal to batch size.

        Args:
            state (StateVector | StateVectorBatched):
                State vector to be updated.

            params (Mapping[str, float] | Mapping[str, Sequence[float]] | None):
                Parameter dictionary. Use float values for `StateVector` and sequences for `StateVectorBatched`.

            classical_register (ClassicalRegister | ClassicalRegisterBatched | None):
                Classical register to be used by gates in the circuit.

            seed (int | None):
                Seed for random number generator.
        """

    @overload
    def update_quantum_state(self, state: StateVector, **kwargs) -> None: ...

    @overload
    def update_quantum_state(self, state: StateVector, params: Mapping[str, float] = {}, seed: int | None = None) -> None: ...

    @overload
    def update_quantum_state(self, state: StateVector, classical_register: scaluq.scaluq_core.ClassicalRegister, params: Mapping[str, float] = {}, seed: int | None = None) -> None: ...

    @overload
    def update_quantum_state(self, state: StateVector, classical_register: scaluq.scaluq_core.ClassicalRegister, **kwargs) -> None: ...

    @overload
    def update_quantum_state(self, state: StateVectorBatched, **kwargs) -> None: ...

    @overload
    def update_quantum_state(self, state: StateVectorBatched, params: Mapping[str, Sequence[float]] = {}, seed: int | None = None) -> None: ...

    @overload
    def update_quantum_state(self, state: StateVectorBatched, classical_register: scaluq.scaluq_core.ClassicalRegisterBatched, params: Mapping[str, Sequence[float]] = {}, seed: int | None = None) -> None: ...

    @overload
    def update_quantum_state(self, state: StateVectorBatched, classical_register: scaluq.scaluq_core.ClassicalRegisterBatched, **kwargs) -> None: ...

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVector, **kwargs) -> None:
        """
        Apply gate to the StateVector. StateVector in args is directly updated. If the circuit contains parametric gate, you have to give real value of parameter as "name=value" format in kwargs.
        """

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVector, params: Mapping[str, float] = {}, seed: int | None = None) -> None:
        """
        Apply gate to the StateVector. StateVector in args is directly updated. If the circuit contains parametric gate, you have to give real value of parameter as dict[str, float] in 2nd arg.
        """

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVector, classical_register: scaluq.scaluq_core.ClassicalRegister, params: Mapping[str, float] = {}, seed: int | None = None) -> None:
        """
        Apply gate to the StateVector with classical register and optional seed.
        """

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVector, classical_register: scaluq.scaluq_core.ClassicalRegister, **kwargs) -> None:
        """
        Apply gate to the StateVector with classical register. If the circuit contains parametric gate, give parameter values as "name=value" in kwargs.
        """

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVectorBatched, **kwargs) -> None:
        """
        Apply gate to the StateVectorBatched. StateVectorBatched in args is directly updated. If the circuit contains parametric gate, you have to give real value of parameter as "name=[value1, value2, ...]" format in kwargs.
        """

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVectorBatched, params: Mapping[str, Sequence[float]] = {}, seed: int | None = None) -> None:
        """
        Apply gate to the StateVectorBatched. StateVectorBatched in args is directly updated. If the circuit contains parametric gate, you have to give real value of parameter as dict[str, list[float]] in 2nd arg.
        """

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVectorBatched, classical_register: scaluq.scaluq_core.ClassicalRegisterBatched, params: Mapping[str, Sequence[float]] = {}, seed: int | None = None) -> None:
        """
        Apply gate to the StateVectorBatched with classical register and optional seed.
        """

    @overload
    def update_quantum_state(self, state: scaluq.scaluq_core.host_serial.f64.StateVectorBatched, classical_register: scaluq.scaluq_core.ClassicalRegisterBatched, **kwargs) -> None:
        """
        Apply gate to the StateVectorBatched with classical register. If the circuit contains parametric gate, give parameter values as "name=[value1, value2, ...]" in kwargs.
        """

    @overload
    def optimize(self, max_block_size: int = 3) -> None: ...

    @overload
    def optimize(self, max_block_size: int = 3) -> None:
        """
        Optimize circuit. Create qubit dependency tree and merge neighboring gates if the new gate has less than or equal to `max_block_size` or the new gate is Pauli.
        """

    @overload
    def simulate_noise(self, initial_state: StateVector, sampling_count: int, parameters: Mapping[str, float] = {}, seed: int | None = None) -> list[tuple[StateVector, int]]: ...

    @overload
    def simulate_noise(self, initial_state: scaluq.scaluq_core.host_serial.f64.StateVector, sampling_count: int, parameters: Mapping[str, float] = {}, seed: int | None = None) -> list[tuple[scaluq.scaluq_core.host_serial.f64.StateVector, int]]:
        """
        Simulate noise circuit. Return all the possible states and their counts.
        """

    @overload
    def compute_expectation_gradient_backprop(self, state: StateVector, bistate: StateVector, parameters: Mapping[str, float]) -> dict[str, float]: ...

    @overload
    def compute_expectation_gradient_backprop(self, state: scaluq.scaluq_core.host_serial.f64.StateVector, bistate: scaluq.scaluq_core.host_serial.f64.StateVector, parameters: Mapping[str, float]) -> dict[str, float]:
        """
        Low-level implementation for expectation gradient that assumes the forward state and observable-applied bistate are already prepared, and computes gradient using back propagation.
        """

    @overload
    def compute_expectation_gradient(self, observable: Operator, parameters: Mapping[str, float]) -> dict[str, float]: ...

    @overload
    def compute_expectation_gradient(self, observable: scaluq.scaluq_core.host_serial.f64.Operator, parameters: Mapping[str, float]) -> dict[str, float]:
        """
        Compute gradient of expectation value of observable using back propagation.
        """

    @overload
    def add_observable_rotation_gate(self, observable: Operator, angle: float, num_repeats: int) -> None: ...

    @overload
    def add_observable_rotation_gate(self, observable: scaluq.scaluq_core.host_serial.f64.Operator, angle: float, num_repeats: int) -> None:
        """
        Add observable rotation gate.

        Args:
            observable (Operator):
                observable

            angle (float):
                angle

            num_repeats (int):
                repeats num

        Examples:
            >>> circuit = Circuit()
            >>> terms = []
            >>> terms.append(PauliOperator("Z 0 Z 1"))
            >>> observable = Operator(terms)
            >>> circuit.add_observable_rotation_gate(observable, 0.1, 100)
        """

class PauliOperator:
    """
    Pauli operator as coef and tensor product of single pauli for each qubit.

    Examples:
        >>> pauli = PauliOperator("X 3 Y 2")
        >>> print(pauli.to_json())
        {"coef":{"imag":0.0,"real":1.0},"pauli_string":"Y 2 X 3"}
    """

    @overload
    def __init__(self, coef: complex = 1.0) -> None: ...

    @overload
    def __init__(self, target_qubit_list: Sequence[int], pauli_id_list: Sequence[int], coef: complex = 1.0) -> None: ...

    @overload
    def __init__(self, pauli_string: str, coef: complex = 1.0) -> None: ...

    @overload
    def __init__(self, pauli_id_par_qubit: Sequence[int], coef: complex = 1.0) -> None: ...

    @overload
    def __init__(self, bit_flip_mask: int, phase_flip_mask: int, coef: complex = 1.0) -> None: ...

    def coef(self) -> complex:
        """Get property `coef`."""

    def target_qubit_list(self) -> list[int]:
        """Get qubits to be applied pauli."""

    def pauli_id_list(self) -> list[int]:
        """
        Get pauli id to be applied. The order is correspond to the result of `target_qubit_list`
        """

    def get_XZ_mask_representation(self) -> tuple[int, int]:
        """
        Get single-pauli property as binary integer representation. See description of `__init__(bit_flip_mask_py: int, phase_flip_mask_py: int, coef: float=1.)` for details.
        """

    def get_pauli_string(self) -> str:
        """
        Get single-pauli property as string representation. See description of `__init__(pauli_string: str, coef: float=1.)` for details.
        """

    def get_dagger(self) -> PauliOperator:
        """Get adjoint operator."""

    def add_single_pauli(self, target_qubit: int, pauli_id: int) -> None:
        """Add single pauli."""

    def apply_to_state(self, state: StateVector) -> None:
        """Apply pauli to state vector."""

    @overload
    def get_expectation_value(self, state: StateVector) -> complex:
        """
        Get expectation value of measuring state vector. $\\bra{\\psi}P\\ket{\\psi}$.
        """

    @overload
    def get_expectation_value(self, states: StateVectorBatched) -> list[complex]:
        """
        Get expectation value of measuring state vectors. $\\bra{\\psi}P\\ket{\\psi}$.
        """

    @overload
    def get_transition_amplitude(self, source: StateVector, target: StateVector) -> complex:
        """
        Get transition amplitude of measuring state vector. $\\bra{\\chi}P\\ket{\\psi}$.
        """

    @overload
    def get_transition_amplitude(self, states_source: StateVectorBatched, states_target: StateVectorBatched) -> list[complex]:
        """
        Get transition amplitude of measuring state vectors. $\\bra{\\chi}P\\ket{\\psi}$.
        """

    def get_matrix(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the PauliOperator. Tensor product is applied from $(n-1)$ -th qubit to $0$ -th qubit. Only the X, Y, and Z components are taken into account in the result.
        """

    def get_full_matrix(self, n_qubits: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the PauliOperator. Tensor product is applied from $(n-1)$ -th qubit to $0$ -th qubit.
        """

    def get_matrix_ignoring_coef(self) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the PauliOperator, but with forcing `coef=1.`Only the X, Y, and Z components are taken into account in the result.
        """

    def get_full_matrix_ignoring_coef(self, n_qubits: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='C')]:
        """
        Get matrix representation of the PauliOperator, but with forcing `coef=1.`
        """

    def get_matrix_triplets_ignoring_coef(self) -> list["Eigen::Triplet<std::complex<double>, int>"]:
        """
        Get matrix representation of the PauliOperator, but with forcing `coef=1.` and non zero elements
        """

    @overload
    def __mul__(self, arg: PauliOperator, /) -> PauliOperator: ...

    @overload
    def __mul__(self, arg: complex, /) -> PauliOperator: ...

    def to_json(self) -> str:
        """Information as json style."""

    def load_json(self, json_str: str) -> None:
        """Read an object from the JSON representation of the Pauli operator."""

    def to_string(self) -> str:
        """Get string representation of the Pauli operator."""

    def __str__(self) -> str:
        """Get string representation of the Pauli operator."""
