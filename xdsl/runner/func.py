from typing import cast
from traitlets import Any
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import Call, FuncOp, Return
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.ir.core import Block
from xdsl.utils.exceptions import InterpretationError


@register_impls
class FuncFunctions(InterpreterFunctions):
    @impl(FuncOp)
    def run_func(self, interpreter: Interpreter, op: FuncOp, args: tuple[Any, ...]):
        return ()

    @impl(Call)
    def run_call(self, interpreter: Interpreter, op: Call, args: tuple[Any, ...]):
        root = op
        while (nroot := root.parent_op()) is not None:
            root = nroot
        callee_op = None
        for f in root.walk():
            if isinstance(f, FuncOp):
                if f.sym_name.data == op.callee.root_reference.data:
                    callee_op = f
        if callee_op is None:
            raise InterpretationError(f"Didn't find @{op.callee.string_value()}")
        # print(f"Calling {op.callee}({','.join(str(a) for a in args)})")
        ret = interpreter.run_ssacfg_region(callee_op.body, args)
        # print(f"Returning {ret}")
        return ret

    @impl(Return)
    def run_return(self, interpreter: Interpreter, op: Return, args: tuple[Any, ...]):
        return (None, args)
