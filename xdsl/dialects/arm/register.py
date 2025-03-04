from __future__ import annotations

import abc

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import irdl_attr_definition


class ARMRegisterType(RegisterType, abc.ABC):
    """
    The abstract class for all ARM register types.
    """

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
    def instruction_set_name(cls) -> str:
        return "arm"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return ARM_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_"


UNALLOCATED_INT = IntRegisterType.unallocated()
X0 = IntRegisterType.from_spelling("x0")
X1 = IntRegisterType.from_spelling("x1")
X2 = IntRegisterType.from_spelling("x2")
X3 = IntRegisterType.from_spelling("x3")
X4 = IntRegisterType.from_spelling("x4")
X5 = IntRegisterType.from_spelling("x5")
X6 = IntRegisterType.from_spelling("x6")
X7 = IntRegisterType.from_spelling("x7")
X8 = IntRegisterType.from_spelling("x8")
X9 = IntRegisterType.from_spelling("x9")
X10 = IntRegisterType.from_spelling("x10")
X11 = IntRegisterType.from_spelling("x11")
X12 = IntRegisterType.from_spelling("x12")
X13 = IntRegisterType.from_spelling("x13")
X14 = IntRegisterType.from_spelling("x14")
X15 = IntRegisterType.from_spelling("x15")
X16 = IntRegisterType.from_spelling("x16")
X17 = IntRegisterType.from_spelling("x17")
X18 = IntRegisterType.from_spelling("x18")
X19 = IntRegisterType.from_spelling("x19")
X20 = IntRegisterType.from_spelling("x20")
X21 = IntRegisterType.from_spelling("x21")
X22 = IntRegisterType.from_spelling("x22")
X23 = IntRegisterType.from_spelling("x23")
X24 = IntRegisterType.from_spelling("x24")
X25 = IntRegisterType.from_spelling("x25")
X26 = IntRegisterType.from_spelling("x26")
X27 = IntRegisterType.from_spelling("x27")
X28 = IntRegisterType.from_spelling("x28")
X29 = IntRegisterType.from_spelling("x29")
X30 = IntRegisterType.from_spelling("x30")
