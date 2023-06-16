from traitlets import Any
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls


@register_impls
class BuiltinFunctions(InterpreterFunctions):
    @impl(ModuleOp)
    def run_module(
        self, interpreter: Interpreter, op: ModuleOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        for child_op in op.ops:
            interpreter.run_op(child_op)
        return ()
