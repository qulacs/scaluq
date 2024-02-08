from enum import Enum
from typing import (Any, Callable, Iterable, Optional, Sequence, Typing, Union,
                    overload)

import qulacs2023

def Haar_random_state(arg: int, /) -> qulacs2023.qulacs_core.StateVector:
    """
    Haar_random_state(arg: int, /) -> qulacs2023.qulacs_core.StateVector
    """
    ...

@overload
def Haar_random_state(arg0: int, arg1: int, /) -> qulacs2023.qulacs_core.StateVector:
    """
    Haar_random_state(arg0: int, arg1: int, /) -> qulacs2023.qulacs_core.StateVector
    """
    ...

class InitializationSettings:
    """
    None
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
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_disable_warnings(
        self, arg: bool, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_map_device_id_by(
        self, arg: str, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_num_threads(
        self, arg: int, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_print_configuration(
        self, arg: bool, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_tools_args(
        self, arg: str, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_tools_help(
        self, arg: bool, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_tools_libs(
        self, arg: str, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...
    def set_tune_internals(
        self, arg: bool, /
    ) -> qulacs2023.qulacs_core.InitializationSettings: ...

class StateVector:
    """
    None
    """

    def __init__(self, arg: qulacs2023.qulacs_core.StateVector) -> None:
        """
        __init__(self, arg: qulacs2023.qulacs_core.StateVector) -> None
        """
        ...

    @overload
    def __init__(self) -> None:
        """
        __init__(self) -> None
        """
        ...

    @overload
    def __init__(self, arg: int, /) -> None:
        """
        __init__(self, arg: int, /) -> None
        """
        ...

    def add_state_vector(self, arg: qulacs2023.qulacs_core.StateVector, /) -> None: ...
    def add_state_vector_with_coef(
        self, arg0: complex, arg1: qulacs2023.qulacs_core.StateVector, /
    ) -> None: ...
    def amplitudes(self) -> list[complex]: ...
    def compute_squared_norm(self) -> float: ...
    def dim(self) -> int: ...
    def get_entropy(self) -> float: ...
    def get_marginal_probability(self, arg: list[int], /) -> float: ...
    def get_zero_probability(self, arg: int, /) -> float: ...
    def load(self, arg: list[complex], /) -> None: ...
    def multiply_coef(self, arg: complex, /) -> None: ...
    def n_qubits(self) -> int: ...
    def normalize(self) -> None: ...
    def sampling(self, arg0: int, arg1: int, /) -> list[int]: ...
    def set_computational_basis(self, arg: int, /) -> None: ...
    def set_zero_norm_state(self) -> None: ...
    def set_zero_state(self) -> None: ...
    def to_string(self) -> str: ...

def finalize() -> None: ...
def initialize(
    settings: qulacs2023.qulacs_core.InitializationSettings = ...,
) -> None: ...
