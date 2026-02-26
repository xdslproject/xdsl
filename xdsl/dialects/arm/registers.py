from __future__ import annotations

import abc

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import irdl_attr_definition


class ARMRegisterType(RegisterType, abc.ABC):
    """
    The abstract class for all ARM register types.
    """


ARM_INDEX_BY_NAME = {f"x{i}": i for i in range(0, 31)}


@irdl_attr_definition
class IntRegisterType(ARMRegisterType):
    """
    A scalar ARM register type representing general-purpose integer registers.
    """

    name = "arm.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return ARM_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_"


UNALLOCATED_INT = IntRegisterType.unallocated()
X0 = IntRegisterType.from_name("x0")
X1 = IntRegisterType.from_name("x1")
X2 = IntRegisterType.from_name("x2")
X3 = IntRegisterType.from_name("x3")
X4 = IntRegisterType.from_name("x4")
X5 = IntRegisterType.from_name("x5")
X6 = IntRegisterType.from_name("x6")
X7 = IntRegisterType.from_name("x7")
X8 = IntRegisterType.from_name("x8")
X9 = IntRegisterType.from_name("x9")
X10 = IntRegisterType.from_name("x10")
X11 = IntRegisterType.from_name("x11")
X12 = IntRegisterType.from_name("x12")
X13 = IntRegisterType.from_name("x13")
X14 = IntRegisterType.from_name("x14")
X15 = IntRegisterType.from_name("x15")
X16 = IntRegisterType.from_name("x16")
X17 = IntRegisterType.from_name("x17")
X18 = IntRegisterType.from_name("x18")
X19 = IntRegisterType.from_name("x19")
X20 = IntRegisterType.from_name("x20")
X21 = IntRegisterType.from_name("x21")
X22 = IntRegisterType.from_name("x22")
X23 = IntRegisterType.from_name("x23")
X24 = IntRegisterType.from_name("x24")
X25 = IntRegisterType.from_name("x25")
X26 = IntRegisterType.from_name("x26")
X27 = IntRegisterType.from_name("x27")
X28 = IntRegisterType.from_name("x28")
X29 = IntRegisterType.from_name("x29")
X30 = IntRegisterType.from_name("x30")
