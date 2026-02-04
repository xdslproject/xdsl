from __future__ import annotations

from typing import Any

from xdsl.dialects import rv32
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.riscv import RiscvFunctions


@register_impls
class Rv32Functions(InterpreterFunctions):
    @impl(rv32.LiOp)
    def run_li(
        self,
        interpreter: Interpreter,
        op: rv32.LiOp,
        args: tuple[Any, ...],
    ):
        results = (RiscvFunctions.get_immediate_value(interpreter, op.immediate),)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)
