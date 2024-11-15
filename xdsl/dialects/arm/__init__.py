"""
ARM dialect, based on the ISA specification in:
https://developer.arm.com/documentation/102374/0101/Overview
"""

from xdsl.ir import Dialect

from .register import IntRegisterType

ARM = Dialect(
    "arm",
    [],
    [
        IntRegisterType,
    ],
)
