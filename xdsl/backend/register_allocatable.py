import abc
from collections.abc import Sequence
from typing import NamedTuple

from xdsl.ir import SSAValue


class RegisterConstraints(NamedTuple):
    """
    Values used by an instruction.
    A collection of operations in `inouts` represents the constraint that they must be
    allocated to the same register.
    """

    ins: Sequence[SSAValue]
    outs: Sequence[SSAValue]
    inouts: Sequence[Sequence[SSAValue]]


class HasRegisterConstraints(abc.ABC):
    @abc.abstractmethod
    def get_register_constraints(self) -> RegisterConstraints:
        """
        The values with register types used by this operation, for use in register
        allocation.
        """
        raise NotImplementedError()
