"""
ARM dialect, based on the ISA [specification](https://developer.arm.com/documentation/102374/0101/Overview).
"""

from typing import IO

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Dialect

from .ops import CmpRegOp, DSMovOp, DSSMulOp, GetRegisterOp, LabelOp
from .registers import IntRegisterType


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    printer = AssemblyPrinter(stream=output)
    printer.print_module(module)


ARM = Dialect(
    "arm",
    [
        GetRegisterOp,
        CmpRegOp,
        DSMovOp,
        DSSMulOp,
        LabelOp,
    ],
    [
        IntRegisterType,
    ],
)
