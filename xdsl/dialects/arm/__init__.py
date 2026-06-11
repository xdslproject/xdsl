"""
ARM dialect, based on the ISA [specification](https://developer.arm.com/documentation/102374/0101/Overview).
"""

from dataclasses import dataclass
from typing import IO

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Dialect
from xdsl.utils.target import Target

from .ops import CmpRegOp, DSMovOp, DSSMulOp, GetRegisterOp, LabelOp
from .registers import IntRegisterType


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    printer = AssemblyPrinter(stream=output)
    printer.print_module(module)


@dataclass(frozen=True)
class ARMAsmTarget(Target):
    name = "arm-asm"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        print_assembly(module, output)


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
