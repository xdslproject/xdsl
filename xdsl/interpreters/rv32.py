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
    @impl(rv32.SlliOp)
    def run_shift_left_i(
        self,
        interpreter: Interpreter,
        op: rv32.SlliOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        imm = RiscvFunctions.get_immediate_value(interpreter, op.immediate)
        assert isinstance(imm, int)
        results = (args[0] << imm,)
        return RiscvFunctions.set_reg_values(interpreter, op.results, results)
