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
        root = op.parent_op()
        while (nroot := root.parent_op()) is not None:
            root = nroot
        for f in root.walk():
            if isinstance(f, FuncOp):
                if f.sym_name.data == op.callee.root_reference.data:
                    interpreter.push_scope()
                    interpreter.set_values(zip(f.body.blocks[0].args, args))
                    for instruction in f.body.ops:
                        interpreter.run(instruction)
                    return interpreter.get_values(
                        cast(Return, f.body.ops.last).operands
                    )
                else:
                    print(f.sym_name, op.callee.root_reference)
        raise InterpretationError(f"Didn't find @{op.callee.string_value()}")

    @impl(Return)
    def run_return(
        self, interpreter: Interpreter, op: Return, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return ()
