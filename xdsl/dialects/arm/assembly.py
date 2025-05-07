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


class square_brackets_reg(AssemblyInstructionArg):
    """
    A wrapper around SSAValue to be printed in assembly.
    Only valid if the type of the value is a RegisterType.
    This class handles the case where a register contains a pointer reference,
    and therefore should be printed within square brackets.
    """

    value: SSAValue

    def __init__(self, value: SSAValue) -> None:
        self.value = value

    def assembly_str(self) -> str:
        assert isinstance(self.value.type, RegisterType)
        return f"[{self.value.type.register_name.data}]"
