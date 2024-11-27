from dataclasses import dataclass
from typing import Any

from xdsl.dialects import riscv_debug
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.riscv import RiscvFunctions


@dataclass
@register_impls
class RiscvDebugFunctions(InterpreterFunctions):
    @impl(riscv_debug.PrintfOp)
    def run_printf(
        self,
        interpreter: Interpreter,
        op: riscv_debug.PrintfOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        print(op.format_str.data.format(*args), end="", file=interpreter.file)
        return ()
