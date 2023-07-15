from typing import Any

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls


@register_impls
class BuiltinFunctions(InterpreterFunctions):
    @impl(UnrealizedConversionCastOp)
    def run_cast(
        self,
        interpreter: Interpreter,
        op: UnrealizedConversionCastOp,
        args: tuple[Any, ...],
    ):
        return tuple(
            interpreter.cast_value(o.type, r.type, arg)
            for (o, r, arg) in zip(op.operands, op.results, args)
        )
