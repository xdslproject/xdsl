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
    is_ptr_to_mem: bool | None = False

    def __init__(self, value: SSAValue, *, is_ptr_to_mem: bool = False) -> None:
        self.value = value
        self.is_ptr_to_mem = is_ptr_to_mem

    def assembly_str(self) -> str:
        assert isinstance(self.value.type, RegisterType)
        if self.is_ptr_to_mem:
            return f"[{self.value.type.register_name.data}]"
        else:
            return self.value.type.register_name.data
