from enum import Enum
from typing import (Any, Callable, Iterable, Optional, Sequence, Typing, Union,
                    overload)

import scaluq

def CNot(arg0: int, arg1: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of CX.
    [note] CNot is an alias of CX.
    """
    ...

def CX(arg0: int, arg1: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of CX.
    """
    ...

class CXGate:
    """
    Specific class of single-qubit-controlled Pauli-X gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def control(self) -> int:
        """
        Get property `control`.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int:
        """
        Get property `target`.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def CZ(arg0: int, arg1: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of CZ.
    """
    ...

class CZGate:
    """
    Specific class of single-qubit-controlled Pauli-Z gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def control(self) -> int:
        """
        Get property `control`.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int:
        """
        Get property `target`.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class Circuit:
    """
    Quantum circuit represented as gate array
    """

    def __init__(self, arg: int, /) -> None:
        """
        Initialize empty circuit of specified qubits.
        """
        ...

    def add_circuit(self, arg: scaluq.scaluq_core.Circuit, /) -> None:
        """
        Add all gates in specified circuit. Given gates are copied.
        """
        ...

    def add_gate(self, arg: scaluq.scaluq_core.Gate, /) -> None:
        """
        Add gate. Given gate is copied.
        """
        ...

    def calculate_depth(self) -> int:
        """
        Get depth of circuit.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Circuit:
        """
        Copy circuit. All the gates inside is copied.
        """
        ...

    def gate_count(self) -> int:
        """
        Get property of `gate_count`.
        """
        ...

    def gate_list(self) -> list[scaluq.scaluq_core.Gate]:
        """
        Get property of `gate_list`.
        """
        ...

    def get(self, arg: int, /) -> scaluq.scaluq_core.Gate:
        """
        Get reference of i-th gate.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Circuit:
        """
        Get inverse of circuit. ALl the gates are newly created.
        """
        ...

    def n_qubits(self) -> int:
        """
        Get property of `n_qubits`.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to the StateVector. StateVector in args is directly updated.
        """
        ...

def FusedSwap(arg0: int, arg1: int, arg2: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of FusedSwap.
    """
    ...

class FusedSwapGate:
    """
    Specific class of fused swap gate, which swap qubits in $[\\mathrm{qubit\\_index1},\\mathrm{qubit\\_index1}+\\mathrm{block\\_size})$ and qubits in $[\\mathrm{qubit\\_index2},\\mathrm{qubit\\_index2}+\\mathrm{block\\_size})$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def block_size(self) -> int:
        """
        Get property `block_size`.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def qubit_index1(self) -> int:
        """
        Get property `qubit_index1`.
        """
        ...

    def qubit_index2(self) -> int:
        """
        Get property `qubit_index2`.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class Gate:
    """
    General class of QuantumGate.\n.. note:: Downcast to requred to use gate-specific functions.
    """

    def __init__(self, arg: scaluq.scaluq_core.PauliRotationGate, /) -> None:
        """
        Upcast from `PauliRotationGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.IGate, /) -> None:
        """
        Upcast from `IGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.GlobalPhaseGate, /) -> None:
        """
        Upcast from `GlobalPhaseGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.XGate, /) -> None:
        """
        Upcast from `XGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.YGate, /) -> None:
        """
        Upcast from `YGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.ZGate, /) -> None:
        """
        Upcast from `ZGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.HGate, /) -> None:
        """
        Upcast from `HGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.SGate, /) -> None:
        """
        Upcast from `SGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.SdagGate, /) -> None:
        """
        Upcast from `SdagGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.TGate, /) -> None:
        """
        Upcast from `TGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.TdagGate, /) -> None:
        """
        Upcast from `TdagGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.SqrtXGate, /) -> None:
        """
        Upcast from `SqrtXGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.SqrtXdagGate, /) -> None:
        """
        Upcast from `SqrtXdagGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.SqrtYGate, /) -> None:
        """
        Upcast from `SqrtYGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.SqrtYdagGate, /) -> None:
        """
        Upcast from `SqrtYdagGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.P0Gate, /) -> None:
        """
        Upcast from `P0Gate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.P1Gate, /) -> None:
        """
        Upcast from `P1Gate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.RXGate, /) -> None:
        """
        Upcast from `RXGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.RYGate, /) -> None:
        """
        Upcast from `RYGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.RZGate, /) -> None:
        """
        Upcast from `RZGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.U1Gate, /) -> None:
        """
        Upcast from `U1Gate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.U2Gate, /) -> None:
        """
        Upcast from `U2Gate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.U3Gate, /) -> None:
        """
        Upcast from `U3Gate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.OneQubitMatrixGate, /) -> None:
        """
        Upcast from `OneQubitMatrixGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.CXGate, /) -> None:
        """
        Upcast from `CXGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.CZGate, /) -> None:
        """
        Upcast from `CZGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.SwapGate, /) -> None:
        """
        Upcast from `SwapGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.TwoQubitMatrixGate, /) -> None:
        """
        Upcast from `TwoQubitMatrixGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.FusedSwapGate, /) -> None:
        """
        Upcast from `FusedSwapGate`.
        """
        ...

    @overload
    def __init__(self, arg: scaluq.scaluq_core.PauliGate, /) -> None:
        """
        Upcast from `PauliGate`.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class GateType(Enum):
    """
    <attribute '__doc__' of 'GateType' objects>
    """

    CX: Any

    CZ: Any

    FusedSwap: Any

    GlobalPhase: Any

    H: Any

    I: Any

    OneQubitMatrix: Any

    P0: Any

    P1: Any

    Pauli: Any

    PauliRotation: Any

    RX: Any

    RY: Any

    RZ: Any

    S: Any

    Sdag: Any

    SqrtX: Any

    SqrtXdag: Any

    SqrtY: Any

    SqrtYdag: Any

    Swap: Any

    T: Any

    Tdag: Any

    TwoQubitMatrix: Any

    U1: Any

    U2: Any

    U3: Any

    X: Any

    Y: Any

    Z: Any

def GlobalPhase(arg: float, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of GlobalPhase.
    """
    ...

class GlobalPhaseGate:
    """
    Specific class of gate, which rotate global phase, represented as $e^{i\\mathrm{phase}}I$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def phase(self) -> float:
        """
        Get `phase` property
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def H(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of H.
    """
    ...

class HGate:
    """
    Specific class of Hadamard gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def I() -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of I.
    """
    ...

class IGate:
    """
    Specific class of Pauli-I gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class InitializationSettings:
    """
    Wrapper class of Kokkos's InitializationSettings.\n.. note:: See details: https://kokkos.org/kokkos-core-wiki/API/core/initialize_finalize/InitializationSettings.html
    """

    def __init__(self) -> None: ...
    def get_device_id(self) -> int: ...
    def get_disable_warnings(self) -> bool: ...
    def get_map_device_id_by(self) -> str: ...
    def get_num_threads(self) -> int: ...
    def get_print_configuration(self) -> bool: ...
    def get_tools_args(self) -> str: ...
    def get_tools_help(self) -> bool: ...
    def get_tools_libs(self) -> str: ...
    def get_tune_internals(self) -> bool: ...
    def has_device_id(self) -> bool: ...
    def has_disable_warnings(self) -> bool: ...
    def has_map_device_id_by(self) -> bool: ...
    def has_num_threads(self) -> bool: ...
    def has_print_configuration(self) -> bool: ...
    def has_tools_args(self) -> bool: ...
    def has_tools_help(self) -> bool: ...
    def has_tools_libs(self) -> bool: ...
    def has_tune_internals(self) -> bool: ...
    def set_device_id(
        self, arg: int, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_disable_warnings(
        self, arg: bool, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_map_device_id_by(
        self, arg: str, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_num_threads(
        self, arg: int, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_print_configuration(
        self, arg: bool, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_tools_args(
        self, arg: str, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_tools_help(
        self, arg: bool, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_tools_libs(
        self, arg: str, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...
    def set_tune_internals(
        self, arg: bool, /
    ) -> scaluq.scaluq_core.InitializationSettings: ...

class OneQubitMatrixGate:
    """
    Specific class of single-qubit dense matrix gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def matrix(self) -> list[list[complex]]: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class Operator:
    """
    None
    """

    def __init__(self, arg: int, /) -> None: ...
    def add_operator(self, arg: scaluq.scaluq_core.PauliOperator, /) -> None: ...
    def add_random_operator(
        self, operator_count: int, seed: Optional[int] = None
    ) -> None: ...
    def apply_to_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None: ...
    def get_dagger(self) -> scaluq.scaluq_core.Operator: ...
    def get_expectation_value(
        self, arg: scaluq.scaluq_core.StateVector, /
    ) -> complex: ...
    def get_transition_amplitude(
        self,
        arg0: scaluq.scaluq_core.StateVector,
        arg1: scaluq.scaluq_core.StateVector,
        /,
    ) -> complex: ...
    def is_hermitian(self) -> bool: ...
    def n_qubits(self) -> int: ...
    def optimize(self) -> None: ...
    def terms(self) -> list[scaluq.scaluq_core.PauliOperator]: ...
    def to_string(self) -> str: ...

def P0(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of P0.
    """
    ...

class P0Gate:
    """
    Specific class of projection gate to $\\ket{0}$.\n.. note:: This gate is not unitary.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def P1(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of P1.
    """
    ...

class P1Gate:
    """
    Specific class of projection gate to $\\ket{1}$.\n.. note:: This gate is not unitary.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def Pauli(arg: scaluq.scaluq_core.PauliOperator, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of Pauli.
    """
    ...

class PauliGate:
    """
    Specific class of multi-qubit pauli gate, which applies single-qubit Pauli gate to each of qubit.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class PauliOperator:
    """
    None
    """

    def __init__(
        self, bit_flip_mask: int, phase_flip_mask: int, coef: complex = 1.0
    ) -> None:
        """
        __init__(self, bit_flip_mask: int, phase_flip_mask: int, coef: complex = 1.0) -> None
        """
        ...

    @overload
    def __init__(self, coef: complex = 1.0) -> None:
        """
        __init__(self, coef: complex = 1.0) -> None
        """
        ...

    @overload
    def __init__(
        self,
        target_qubit_list: list[int],
        pauli_id_list: list[int],
        coef: complex = 1.0,
    ) -> None:
        """
        __init__(self, target_qubit_list: list[int], pauli_id_list: list[int], coef: complex = 1.0) -> None
        """
        ...

    @overload
    def __init__(self, pauli_string: str, coef: complex = 1.0) -> None:
        """
        __init__(self, pauli_string: str, coef: complex = 1.0) -> None
        """
        ...

    @overload
    def __init__(self, pauli_id_par_qubit: list[int], coef: complex = 1.0) -> None:
        """
        __init__(self, pauli_id_par_qubit: list[int], coef: complex = 1.0) -> None
        """
        ...

    def add_single_pauli(self, arg0: int, arg1: int, /) -> None: ...
    def apply_to_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None: ...
    def change_coef(self, arg: complex, /) -> None: ...
    def get_XZ_mask_representation(self) -> tuple[int, int]: ...
    def get_coef(self) -> complex: ...
    def get_dagger(self) -> scaluq.scaluq_core.PauliOperator: ...
    def get_expectation_value(
        self, arg: scaluq.scaluq_core.StateVector, /
    ) -> complex: ...
    def get_pauli_id_list(self) -> list[int]: ...
    def get_pauli_string(self) -> str: ...
    def get_qubit_count(self) -> int: ...
    def get_target_qubit_list(self) -> list[int]: ...
    def get_transition_amplitude(
        self,
        arg0: scaluq.scaluq_core.StateVector,
        arg1: scaluq.scaluq_core.StateVector,
        /,
    ) -> complex: ...

def PauliRotation(
    arg0: scaluq.scaluq_core.PauliOperator, arg1: float, /
) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of PauliRotation.
    """
    ...

class PauliRotationGate:
    """
    Specific class of multi-qubit pauli-rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}P}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def RX(arg0: int, arg1: float, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of RX.
    """
    ...

class RXGate:
    """
    Specific class of X rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}X}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def angle(self) -> float:
        """
        Get `angle` property.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def RY(arg0: int, arg1: float, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of RY.
    """
    ...

class RYGate:
    """
    Specific class of Y rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Y}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def angle(self) -> float:
        """
        Get `angle` property.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def RZ(arg0: int, arg1: float, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of RZ.
    """
    ...

class RZGate:
    """
    Specific class of Z rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Z}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def angle(self) -> float:
        """
        Get `angle` property.
        """
        ...

    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def S(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of S.
    """
    ...

class SGate:
    """
    Specific class of S gate, represented as $\\begin{bmatrix}
    1 & 0\\\\
    0 & i
    \\end{bmatrix}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def Sdag(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of Sdag.
    """
    ...

class SdagGate:
    """
    Specific class of inverse of S gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def SqrtX(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of SqrtX.
    """
    ...

class SqrtXGate:
    """
    Specific class of sqrt(X) gate, represented as $\\begin{bmatrix}
    1+i & 1-i\\\\
    1-i & 1+i
    \\end{bmatrix}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def SqrtXdag(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of SqrtXdag.
    """
    ...

class SqrtXdagGate:
    """
    Specific class of inverse of sqrt(X) gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def SqrtY(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of SqrtY.
    """
    ...

class SqrtYGate:
    """
    Specific class of sqrt(Y) gate, represented as $\\begin{bmatrix}
    1+i & -1-i \\\\
    1+i & 1+i
    \\end{bmatrix}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def SqrtYdag(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of SqrtYdag.
    """
    ...

class SqrtYdagGate:
    """
    Specific class of inverse of sqrt(Y) gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class StateVector:
    """
    Vector representation of quantum state.\n.. note:: Qubit index is start from 0. If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ is $b_i$, the state is $\\sum_i b_i 2^i$.
    """

    def Haar_random_state(
        n_qubits: int, seed: Optional[int] = None
    ) -> scaluq.scaluq_core.StateVector:
        """
        Constructing state vector with Haar random state. If seed is not specified, the value from random device is used.
        """
        ...

    def __init__(self, arg: scaluq.scaluq_core.StateVector) -> None:
        """
        Constructing state vector by copying other state.
        """
        ...

    @overload
    def __init__(self, arg: int, /) -> None:
        """
        Construct state vector with specified qubits, initialized with computational basis $\ket{0\dots0}$.
        """
        ...

    def add_state_vector(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Add other state vector and make superposition. $\\ket{\\mathrm{this}} \\leftarrow \\ket{\\mathrm{this}} + \\ket{\\mathrm{state}}$.
        """
        ...

    def add_state_vector_with_coef(
        self, arg0: complex, arg1: scaluq.scaluq_core.StateVector, /
    ) -> None:
        """
        add other state vector with multiplying the coef and make superposition. $\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef}\\ket{\\mathrm{state}}$.
        """
        ...

    def amplitudes(self) -> list[complex]:
        """
        Get all amplitudes with as `list[complex]`.
        """
        ...

    def dim(self) -> int:
        """
        Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).
        """
        ...

    def get_amplitude_at_index(self, arg: int, /) -> complex:
        """
        Get amplitude at one index.\n.. note:: If you want to get all amplitudes, you should use `StateVector::amplitudes()`.
        """
        ...

    def get_entropy(self) -> float:
        """
        Get the entropy of the vector.
        """
        ...

    def get_marginal_probability(self, arg: list[int], /) -> float:
        """
        Get the marginal probability to observe as specified. Specify the result as n-length list. `0` and `1` represent the qubit is observed and get the value. `2` represents the qubit is not observed.
        """
        ...

    def get_squared_norm(self) -> float:
        """
        Get squared norm of the state. $\\braket{\\psi|\\psi}$.
        """
        ...

    def get_zero_probability(self, arg: int, /) -> float:
        """
        Get the probability to observe $\\ket{0}$ at specified index.
        """
        ...

    def load(self, arg: list[complex], /) -> None:
        """
        Load amplitudes of `list[int]` with `dim` length.
        """
        ...

    def multiply_coef(self, arg: complex, /) -> None:
        """
        Multiply coef. $\\ket{\\mathrm{this}}\\leftarrow\\mathrm{coef}\\ket{\\mathrm{this}}$.
        """
        ...

    def n_qubits(self) -> int:
        """
        Get num of qubits.
        """
        ...

    def normalize(self) -> None:
        """
        Normalize state (let $\\braket{\\psi|\\psi} = 1$ by multiplying coef).
        """
        ...

    def sampling(self, sampling_count: int, seed: Optional[int] = None) -> list[int]:
        """
        Sampling specified times. Result is `list[int]` with the `sampling_count` length.
        """
        ...

    def set_amplitude_at_index(self, arg0: int, arg1: complex, /) -> None:
        """
        Manually set amplitude at one index.
        """
        ...

    def set_computational_basis(self, arg: int, /) -> None:
        """
        Initialize with computational basis \\ket{\\mathrm{basis}}.
        """
        ...

    def set_zero_norm_state(self) -> None:
        """
        Initialize with 0 (null vector).
        """
        ...

    def set_zero_state(self) -> None:
        """
        Initialize with computational basis $\\ket{00\\dots0}$.
        """
        ...

    def to_string(self) -> str:
        """
        Information as `str`.
        """
        ...

def Swap(arg0: int, arg1: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of Swap.
    """
    ...

class SwapGate:
    """
    Specific class of two-qubit swap gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target1(self) -> int:
        """
        Get property `target1`.
        """
        ...

    def target2(self) -> int:
        """
        Get property `target2`.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def T(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of T.
    """
    ...

class TGate:
    """
    Specific class of T gate, represented as $\\begin{bmatrix}
    1 & 0\\\\
    0 & e^{i\\pi/4}
    \\end{bmatrix}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def Tdag(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of Tdag.
    """
    ...

class TdagGate:
    """
    Specific class of inverse of T gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

class TwoQubitMatrixGate:
    """
    Specific class of double-qubit dense matrix gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def matrix(self) -> None:
        """
        Get property `matrix`.
        """
        ...

    def target1(self) -> int:
        """
        Get property `target1`.
        """
        ...

    def target2(self) -> int:
        """
        Get property `target2`.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def U1(arg0: int, arg1: float, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of U1.
    """
    ...

class U1Gate:
    """
    Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, represented as $\\begin{bmatrix}
    1 & 0\\\\
    0 & e^{i\\lambda}
    \\end{bmatrix}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def lambda_(self) -> float:
        """
        Get `lambda` property.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def U2(arg0: int, arg1: float, arg2: float, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of U2.
    """
    ...

class U2Gate:
    """
    Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, represented as $\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\
    e^{i\\phi} & e^{i(\\phi+\\lambda)}
    \\end{bmatrix}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def lambda_(self) -> float:
        """
        Get `lambda` property.
        """
        ...

    def phi(self) -> float:
        """
        Get `phi` property.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def U3(arg0: int, arg1: float, arg2: float, arg3: float, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of U3.
    """
    ...

class U3Gate:
    """
    Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, represented as $\\begin{bmatrix}
    \\cos \\frac{\\theta}{2} & -e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\
    e^{i\\phi}\\sin\\frac{\\theta}{2} & e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}
    \\end{bmatrix}$.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def lambda_(self) -> float:
        """
        Get `lambda` property.
        """
        ...

    def phi(self) -> float:
        """
        Get `phi` property.
        """
        ...

    def theta(self) -> float:
        """
        Get `theta` property.
        """
        ...

    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def X(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of X.
    """
    ...

class XGate:
    """
    Specific class of Pauli-X gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def Y(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of Y.
    """
    ...

class YGate:
    """
    Specific class of Pauli-Y gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def Z(arg: int, /) -> scaluq.scaluq_core.Gate:
    """
    Generate general Gate class instance of Z.
    """
    ...

class ZGate:
    """
    Specific class of Pauli-Z gate.\n.. note:: Upcast is required to use gate-general functions (ex: add to Circuit).
    """

    def __init__(self, arg: scaluq.scaluq_core.Gate, /) -> None: ...
    def copy(self) -> scaluq.scaluq_core.Gate:
        """
        Copy gate as `Gate` type.
        """
        ...

    def gate_type(self) -> scaluq.scaluq_core.GateType:
        """
        Get gate type as `GateType` enum.
        """
        ...

    def get_control_qubit_list(self) -> list[int]:
        """
        Get control qubits as `list[int]`.
        """
        ...

    def get_inverse(self) -> scaluq.scaluq_core.Gate:
        """
        Generate inverse gate as `Gate` type. If not exists, return None.
        """
        ...

    def get_target_qubit_list(self) -> list[int]:
        """
        Get target qubits as `list[int]`. **Control qubits is not included.**
        """
        ...

    def target(self) -> int: ...
    def update_quantum_state(self, arg: scaluq.scaluq_core.StateVector, /) -> None:
        """
        Apply gate to `state_vector`. `state_vector` in args is directly updated.
        """
        ...

def finalize() -> None:
    """
    Terminate the Kokkos execution environment. Release the resources.
    """
    ...

def initialize(settings: scaluq.scaluq_core.InitializationSettings = ...) -> None:
    """
    **You must call this before any scaluq function.** Initialize the Kokkos execution environment.
    """
    ...

def is_finalized() -> bool:
    """
    Return true if `finalize()` is already called.
    """
    ...

def is_initialized() -> bool:
    """
    Return true if `initialize()` is already called.
    """
    ...
