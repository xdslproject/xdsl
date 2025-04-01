import abc

from xdsl.backend.register_type import RegisterType
from xdsl.ir import SSAValue


class AssemblyInstructionArg(abc.ABC):
    """
    Abstract base class for arguments to one line of assembly.
    """

    @abc.abstractmethod
    def assembly_str(self) -> str:
        raise NotImplementedError()


class reg(AssemblyInstructionArg):
    """
    A wrapper around SSAValue to be printed in assembly.
    Only valid if the type of the value is a RegisterType.
    """

    value: SSAValue

    def __init__(self, value: SSAValue) -> None:
        self.value = value

    def assembly_str(self) -> str:
        assert isinstance(self.value.type, RegisterType)
        return self.value.type.register_name.data
