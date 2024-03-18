from __future__ import annotations

from collections.abc import Sequence

from xdsl.backend.register_type import RegisterType
from xdsl.ir import Attribute
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.utils.exceptions import VerifyException


class X86RegisterType(RegisterType):
    """
    The abstract class for all x86 register types.
    """

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            name = parser.parse_optional_identifier()
            if name is not None:
                if not name.startswith("e") and not name.startswith("r"):
                    assert name in cls.abi_index_by_name(), f"{name}"
            else:
                name = ""
            return cls._parameters_from_spelling(name)

    def verify(self) -> None:
        name = self.spelling.data
        if not self.is_allocated or name.startswith("e") or name.startswith("r"):
            return
        if name not in type(self).abi_index_by_name():
            raise VerifyException(f"{name} not in {self.instruction_set_name()}")


X86_INDEX_BY_NAME = {
    "rax": 0,
    "rcx": 1,
    "rdx": 2,
    "rbx": 3,
    "rsp": 4,
    "rbp": 5,
    "rsi": 6,
    "rdi": 7,
    "r8": 8,
    "r9": 9,
    "r10": 10,
    "r11": 11,
    "r12": 12,
    "r13": 13,
    "r14": 14,
    "r15": 15,
}


@irdl_attr_definition
class GeneralRegisterType(X86RegisterType):
    """
    A scalar x86 register type representing general registers.
    """

    name = "x86.reg"

    @classmethod
    def unallocated(cls) -> GeneralRegisterType:
        return UNALLOCATED_GENERAL

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return X86_INDEX_BY_NAME


UNALLOCATED_GENERAL = GeneralRegisterType("")
RAX = GeneralRegisterType("rax")
RCX = GeneralRegisterType("rcx")
RDX = GeneralRegisterType("rdx")
RBX = GeneralRegisterType("rbx")
RSP = GeneralRegisterType("rsp")
RBP = GeneralRegisterType("rbp")
RSI = GeneralRegisterType("rsi")
RDI = GeneralRegisterType("rdi")
R8 = GeneralRegisterType("r8")
R9 = GeneralRegisterType("r9")
R10 = GeneralRegisterType("r10")
R11 = GeneralRegisterType("r11")
R12 = GeneralRegisterType("r12")
R13 = GeneralRegisterType("r13")
R14 = GeneralRegisterType("r14")
R15 = GeneralRegisterType("r15")
