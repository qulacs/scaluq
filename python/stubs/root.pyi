import enum
from typing import overload

from scaluq.scaluq_core import (
    default as default,
    host_serial as host_serial
)


def initialize() -> None:
    """
    Initialize the Kokkos execution environment.

    Notes:
        This is automatically called when the program starts. You do not have to call this manually unless you have called `finalize` before.
    """

def finalize() -> None:
    """
    Terminate the Kokkos execution environment. Release the resources.

    Notes:
        Finalization fails if there exists `StateVector` allocated. You must use `StateVector` only inside inner scopes than the usage of `finalize` or delete all of existing `StateVector`.

        This is automatically called when the program exits. If you call this manually, you cannot use most of scaluq's functions until the program exits.
    """

def is_initialized() -> bool:
    """Return true if :func:`~scaluq.initialize()` is already called."""

def is_finalized() -> bool:
    """Return true if :func:`~scaluq.finalize()` is already called."""

def synchronize() -> None:
    """
    Synchronize the device if the execution space is not host.

    Notes:
        This function is required to ensure that all operations on device are finished when you measure the elapsed time of some operations on device.
    """

class ClassicalRegister:
    """Classical register."""

    def __init__(self, register_size: int) -> None:
        """Initialize classical register."""

    def register_size(self) -> int:
        """Get register size."""

    def empty(self) -> bool:
        """Get register binary."""

    def __len__(self) -> int:
        """Get register size."""

    def __getitem__(self, index: int) -> bool:
        """Get classical bit."""

    def __setitem__(self, index: int, value: bool) -> None:
        """Set classical bit."""

    def reset(self) -> None:
        """Reset all bits to `False`."""

class ClassicalRegisterBatched:
    """Batched classical register."""

    def __init__(self, register_size: int, batch_size: int) -> None:
        """Initialize batched classical register."""

    def register_size(self) -> int:
        """Get register size."""

    def batch_size(self) -> int:
        """Get batch size."""

    def __len__(self) -> int:
        """Get batch size."""

    @overload
    def __getitem__(self, batch_index: int) -> ClassicalRegister:
        """Get classical register at `batch_index`."""

    @overload
    def __getitem__(self, idx: tuple) -> bool: ...

    @overload
    def __setitem__(self, batch_index: int, value: ClassicalRegister) -> None:
        """Set classical register at `batch_index`."""

    @overload
    def __setitem__(self, idx: tuple, value: bool) -> None: ...

    def reset(self) -> None:
        """Reset all bits to `False`."""

class GateType(enum.Enum):
    """Enum of Gate Type."""

    I = 1

    GlobalPhase = 2

    X = 3

    Y = 4

    Z = 5

    H = 6

    S = 7

    Sdag = 8

    T = 9

    Tdag = 10

    SqrtX = 11

    SqrtXdag = 12

    SqrtY = 13

    SqrtYdag = 14

    P0 = 15

    P1 = 16

    Measurement = 17

    RX = 18

    RY = 19

    RZ = 20

    U1 = 21

    U2 = 22

    U3 = 23

    Swap = 24

    Ecr = 25

    Pauli = 27

    PauliRotation = 28

    SparseMatrix = 29

    DenseMatrix = 30

    Probabilistic = 31

class ParamGateType(enum.Enum):
    """Enum of ParamGate Type."""

    ParamRX = 1

    ParamRY = 2

    ParamRZ = 3

    ParamPauliRotation = 4

def get_default_execution_space() -> str:
    """
    Get the default execution space.

    Returns:
        str:
            the default execution space, `cuda` or `host`

    Examples:
        >>> get_default_execution_space() # doctest: +SKIP
        'cuda'
    """

def precision_available(arg: str, /) -> bool:
    """
    Return the precision is supported.

    Args:
        precision (str):
            precision name

            This must be one of `f16` `f32` `f64` `bf16`.

    Returns:
        bool:
            the precision is supported

    Examples:
        >>> precision_available('f64') # doctest: +SKIP
        True
        >>> precision_available('bf16') # doctest: +SKIP
        False
    """
