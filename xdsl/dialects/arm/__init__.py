"""
ARM dialect, based on the ISA specification in:
https://developer.arm.com/documentation/102374/0101/Overview
"""

from typing import IO

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Dialect

from .ops import ARMOperation, DSMovOp, DSSMulOp, GetRegisterOp
from .register import IntRegisterType


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        assert isinstance(op, ARMOperation), f"{op}"
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)


ARM = Dialect(
    "arm",
    [
        GetRegisterOp,
        DSMovOp,
        DSSMulOp,
    ],
    [
        IntRegisterType,
    ],
)
