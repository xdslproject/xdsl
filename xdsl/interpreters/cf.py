from typing import Any

from xdsl.dialects import cf
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    Successor,
    impl_terminator,
    register_impls,
)


@register_impls
class CfFunctions(InterpreterFunctions):
    @impl_terminator(cf.BranchOp)
    def run_br(self, interpreter: Interpreter, op: cf.BranchOp, args: tuple[Any, ...]):
        return Successor(op.successor, args), ()

    @impl_terminator(cf.ConditionalBranchOp)
    def run_cond_br(
        self,
        interpreter: Interpreter,
        op: cf.ConditionalBranchOp,
        args: tuple[Any, ...],
    ):
        cond: int = args[0]
        if cond:
            block_args = interpreter.get_values(op.then_arguments)
            return Successor(op.then_block, block_args), ()
        else:
            block_args = interpreter.get_values(op.else_arguments)
            return Successor(op.else_block, block_args), ()
