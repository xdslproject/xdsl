from __future__ import annotations

import abc

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import irdl_attr_definition


class ARMRegisterType(RegisterType, abc.ABC):
    """
    The abstract class for all ARM register types.
    """


ARM_INDEX_BY_NAME = {f"x{i}": i for i in range(0, 31)}

ARM_NEON_INDEX_BY_NAME = {f"v{i}": i for i in range(0, 32)}


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
    def index_by_name(cls) -> dict[str, int]:
        return ARM_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_"


@irdl_attr_definition
class NEONRegisterType(ARMRegisterType):
    """
    A NEON ARM register type.
    """

    name = "arm_neon.neonreg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "arm_neon"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return ARM_NEON_INDEX_BY_NAME

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

UNALLOCATED_NEON = NEONRegisterType.unallocated()
V0 = NEONRegisterType.from_name("v0")
V1 = NEONRegisterType.from_name("v1")
V2 = NEONRegisterType.from_name("v2")
V3 = NEONRegisterType.from_name("v3")
V4 = NEONRegisterType.from_name("v4")
V5 = NEONRegisterType.from_name("v5")
V6 = NEONRegisterType.from_name("v6")
V7 = NEONRegisterType.from_name("v7")
V8 = NEONRegisterType.from_name("v8")
V9 = NEONRegisterType.from_name("v9")
V10 = NEONRegisterType.from_name("v10")
V11 = NEONRegisterType.from_name("v11")
V12 = NEONRegisterType.from_name("v12")
V13 = NEONRegisterType.from_name("v13")
V14 = NEONRegisterType.from_name("v14")
V15 = NEONRegisterType.from_name("v15")
V16 = NEONRegisterType.from_name("v16")
V17 = NEONRegisterType.from_name("v17")
V18 = NEONRegisterType.from_name("v18")
V19 = NEONRegisterType.from_name("v19")
V20 = NEONRegisterType.from_name("v20")
V21 = NEONRegisterType.from_name("v21")
V22 = NEONRegisterType.from_name("v22")
V23 = NEONRegisterType.from_name("v23")
V24 = NEONRegisterType.from_name("v24")
V25 = NEONRegisterType.from_name("v25")
V26 = NEONRegisterType.from_name("v26")
V27 = NEONRegisterType.from_name("v27")
V28 = NEONRegisterType.from_name("v28")
V29 = NEONRegisterType.from_name("v29")
V30 = NEONRegisterType.from_name("v30")
V31 = NEONRegisterType.from_name("v31")
