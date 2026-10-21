from dataclasses import dataclass
from typing import Any

from xdsl.dialects import riscv_debug
from xdsl.dialects.builtin import FloatAttr
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.utils.hints import isa


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
        args = tuple(arg.value.data if isa(arg, FloatAttr) else arg for arg in args)
        print(op.format_str.data.format(*args), end="", file=interpreter.file)
        return ()
