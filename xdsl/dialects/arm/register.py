from __future__ import annotations

from collections.abc import Sequence

from xdsl.backend.register_type import RegisterType
from xdsl.ir import Attribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.utils.exceptions import VerifyException


class ARMRegisterType(RegisterType):
    """
    The abstract class for all ARM register types.
    """

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<") is not None:
            name = parser.parse_identifier()
            parser.parse_punctuation(">")
            if not name.startswith("x"):  # only including x regs for now
                assert (
                    name in cls.abi_index_by_name()
                ), f"{name}"  # abi_index_by_name raises NotImplementedError (?)
        else:
            name = ""
        return cls._parameters_from_spelling(name)

    def verify(self) -> None:
        name = self.spelling.data
        if not self.is_allocated or name.startswith("x"):
            return
        if name not in type(self).abi_index_by_name():
            raise VerifyException(f"{name} not in {self.instruction_set_name()}")


ARM_INDEX_BY_NAME = {
    "x0": 0,
    "x1": 1,
    "x2": 2,
    "x3": 3,
    "x4": 4,
    "x5": 5,
    "x6": 6,
    "x7": 7,
    "x8": 8,
    "x9": 9,
    "x10": 10,
    "x11": 11,
    "x12": 12,
    "x13": 13,
    "x14": 14,
    "x15": 15,
    "x16": 16,
    "x17": 17,
    "x18": 18,
    "x19": 19,
    "x20": 20,
    "x21": 21,
    "x22": 22,
    "x23": 23,
    "x24": 24,
    "x25": 25,
    "x26": 26,
    "x27": 27,
    "x28": 28,
    "x29": 29,
    "x30": 30,
}


@irdl_attr_definition
class GeneralRegisterType(ARMRegisterType):
    """
    A scalar ARM register type representing general registers.
    """

    name = "arm.reg"

    @classmethod
    def unallocated(cls) -> GeneralRegisterType:
        return UNALLOCATED_GENERAL

    @classmethod
    def instruction_set_name(cls) -> str:
        return "arm"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return ARM_INDEX_BY_NAME


UNALLOCATED_GENERAL = GeneralRegisterType("")
X0 = GeneralRegisterType("x0")
X1 = GeneralRegisterType("x1")
X2 = GeneralRegisterType("x2")
X3 = GeneralRegisterType("x3")
X4 = GeneralRegisterType("x4")
X5 = GeneralRegisterType("x5")
X6 = GeneralRegisterType("x6")
X7 = GeneralRegisterType("x7")
X8 = GeneralRegisterType("x8")
X9 = GeneralRegisterType("x9")
X10 = GeneralRegisterType("x10")
X11 = GeneralRegisterType("x11")
X12 = GeneralRegisterType("x12")
X13 = GeneralRegisterType("x13")
X14 = GeneralRegisterType("x14")
X15 = GeneralRegisterType("x15")
X16 = GeneralRegisterType("x16")
X17 = GeneralRegisterType("x17")
X18 = GeneralRegisterType("x18")
X19 = GeneralRegisterType("x19")
X20 = GeneralRegisterType("x20")
X21 = GeneralRegisterType("x21")
X22 = GeneralRegisterType("x22")
X23 = GeneralRegisterType("x23")
X24 = GeneralRegisterType("x24")
X25 = GeneralRegisterType("x25")
X26 = GeneralRegisterType("x26")
X27 = GeneralRegisterType("x27")
X28 = GeneralRegisterType("x28")
X29 = GeneralRegisterType("x29")
X30 = GeneralRegisterType("x30")
