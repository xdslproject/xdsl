from __future__ import annotations

from xdsl.dialects.arm.register import NEON128RegisterType
from xdsl.ir import Dialect

ARM_NEON = Dialect(
    "arm_neon",
    [],
    [
        NEON128RegisterType,
    ],
)
