from __future__ import annotations

from xdsl.dialects.arm.register import NEONRegisterType
from xdsl.ir import Dialect

ARM_NEON = Dialect(
    "arm_neon",
    [],
    [
        NEONRegisterType,
    ],
)
