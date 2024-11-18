from __future__ import annotations

import abc
from collections.abc import Sequence

from xdsl.backend.register_type import RegisterType
from xdsl.ir import Attribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser


class ARMRegisterType(RegisterType, abc.ABC):
    """
    The abstract class for all ARM register types.
    """

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<"):
            name = parser.parse_identifier()
            parser.parse_punctuation(">")
        else:
            name = ""
        return cls._parameters_from_spelling(name)

    def verify(self):
        # No verification for now
        ...


ARM_INDEX_BY_NAME = {f"x{i}": i for i in range(0, 31)}


@irdl_attr_definition
class IntRegisterType(ARMRegisterType):
    """
    A scalar ARM register type representing general-purpose integer registers.
    """

    name = "arm.reg"

    @classmethod
    def unallocated(cls) -> IntRegisterType:
        return UNALLOCATED_INT

    @classmethod
    def instruction_set_name(cls) -> str:
        return "arm"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return ARM_INDEX_BY_NAME


UNALLOCATED_INT = IntRegisterType("")
X0 = IntRegisterType("x0")
X1 = IntRegisterType("x1")
X2 = IntRegisterType("x2")
X3 = IntRegisterType("x3")
X4 = IntRegisterType("x4")
X5 = IntRegisterType("x5")
X6 = IntRegisterType("x6")
X7 = IntRegisterType("x7")
X8 = IntRegisterType("x8")
X9 = IntRegisterType("x9")
X10 = IntRegisterType("x10")
X11 = IntRegisterType("x11")
X12 = IntRegisterType("x12")
X13 = IntRegisterType("x13")
X14 = IntRegisterType("x14")
X15 = IntRegisterType("x15")
X16 = IntRegisterType("x16")
X17 = IntRegisterType("x17")
X18 = IntRegisterType("x18")
X19 = IntRegisterType("x19")
X20 = IntRegisterType("x20")
X21 = IntRegisterType("x21")
X22 = IntRegisterType("x22")
X23 = IntRegisterType("x23")
X24 = IntRegisterType("x24")
X25 = IntRegisterType("x25")
X26 = IntRegisterType("x26")
X27 = IntRegisterType("x27")
X28 = IntRegisterType("x28")
X29 = IntRegisterType("x29")
X30 = IntRegisterType("x30")
