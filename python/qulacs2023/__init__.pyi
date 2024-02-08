from typing import Any, Optional, overload, Typing, Sequence, Iterable, Union, Callable
from enum import Enum
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
def initialize() -> None: ...