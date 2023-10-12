from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects import riscv_scf
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.riscv import RiscvFunctions


@register_impls
@dataclass
class RiscvScfFunctions(InterpreterFunctions):
    bitwidth: int = field(default=32)

    @impl(riscv_scf.ForOp)
    def run_for(
        self,
        interpreter: Interpreter,
        op: riscv_scf.ForOp,
        args: tuple[Any, ...],
    ):
        args = RiscvFunctions.get_reg_values(interpreter, op.operands, args)
        lb, ub, step, *loop_args = args
        loop_args = tuple(loop_args)

        for i in range(lb, ub, step):
            RiscvFunctions.set_reg_value(interpreter, op.body.block.args[0].type, i)
            loop_args = interpreter.run_ssacfg_region(
                op.body, (i, *loop_args), "for_loop"
            )

        return loop_args

    @impl_terminator(riscv_scf.YieldOp)
    def run_yield(
        self, interpreter: Interpreter, op: riscv_scf.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
