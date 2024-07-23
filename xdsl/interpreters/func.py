from typing import Any

from xdsl.dialects import func
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl,
    impl_callable,
    impl_terminator,
    register_impls,
)


@register_impls
class FuncFunctions(InterpreterFunctions):
    @impl_terminator(func.Return)
    def run_return(
        self, interpreter: Interpreter, op: func.Return, args: tuple[Any, ...]
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()

    @impl(func.Call)
    def run_call(
        self, interpreter: Interpreter, op: func.Call, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return interpreter.call_op(op.callee.string_value(), args)

    @impl_callable(func.FuncOp)
    def call_func(
        self, interpreter: Interpreter, op: func.FuncOp, args: tuple[Any, ...]
    ):
        if (first_block := op.body.blocks.first) is None or not first_block.ops:
            return interpreter.call_external(op.sym_name.data, op, args)
        else:
            return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)
