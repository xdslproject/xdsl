from __future__ import annotations

from xdsl.dialects.arm.register import FPSIMDRegisterType
from xdsl.ir import Dialect

ARM_SIMD = Dialect(
    "arm_simd",
    [],
    [
        FPSIMDRegisterType,
    ],
)
