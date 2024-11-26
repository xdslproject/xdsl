"""
ARM FUNC dialect, for instructions related to function handling,
based on the ISA specification in:
https://developer.arm.com/documentation/102374/0101/Overview
"""

from xdsl.ir import Dialect

from .ops import RetOp

ARM_FUNC = Dialect(
    "arm_func",
    [
        RetOp,
    ],
)
