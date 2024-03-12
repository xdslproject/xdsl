from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from typing_extensions import Self

from xdsl.ir import (
    Data,
    Dialect,
    TypeAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


class X86RegisterType(Data[str], TypeAttribute, ABC):
    """
    An x86 register type.
    """

    _unallocated: ClassVar[Self | None] = None

    @classmethod
    def unallocated(cls) -> Self:
        if cls._unallocated is None:
            cls._unallocated = cls("")
        return cls._unallocated

    @property
    def register_name(self) -> str:
        """Returns name if allocated, raises ValueError if not"""
        if not self.is_allocated:
            raise ValueError("Cannot get name for unallocated register")
        return self.data

    @property
    def is_allocated(self) -> bool:
        """Returns true if an x86 register is allocated, otherwise false"""
        return bool(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            name = parser.parse_optional_identifier()
            if name is None:
                return ""
            if not name.startswith("e") and not name.startswith("r"):
                assert name in cls.abi_index_by_name(), f"{name}"
            return name

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data)

    def verify(self) -> None:
        name = self.data
        if not self.is_allocated or name.startswith("e") or name.startswith("r"):
            return
        if name not in type(self).abi_index_by_name():
            raise VerifyException(f"{name} not in {self.instruction_set_name()}")

    @classmethod
    @abstractmethod
    def instruction_set_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        raise NotImplementedError()


@irdl_attr_definition
class GeneralRegisterType(X86RegisterType):
    """
    An x86 register type.
    """

    name = "x86.reg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return GeneralRegisterType.X86_INDEX_BY_NAME

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


class Registers(ABC):
    """Namespace for named register constants."""

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


X86 = Dialect(
    "x86",
    [],
    [
        GeneralRegisterType,
    ],
)
