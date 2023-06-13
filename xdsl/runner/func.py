from typing import cast
from traitlets import Any
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import Call, FuncOp, Return
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError


@register_impls
class FuncFunctions(InterpreterFunctions):
    @impl(FuncOp)
    def run_func(
        self, interpreter: Interpreter, op: FuncOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return ()

    @impl(Call)
    def run_call(
        self, interpreter: Interpreter, op: Call, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert (parent := op.parent_op()) is not None
        for f in parent.walk():
            if isinstance(f, FuncOp) and f.sym_name == StringAttr("main"):
                for instruction in f.body.ops:
                    interpreter.run(instruction)
                return interpreter.get_values(cast(Return, f.body.ops.last).operands)
        raise InterpretationError(f"Didn't find @{op.callee.string_value}")

    @impl(Return)
    def run_return(
        self, interpreter: Interpreter, op: Return, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return ()
