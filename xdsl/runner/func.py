from traitlets import Any
from xdsl.dialects.func import Call, FuncOp, Return
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError


@register_impls
class FuncFunctions(InterpreterFunctions):
    @impl(FuncOp)
    def run_func(self, interpreter: Interpreter, op: FuncOp, args: tuple[Any, ...]):
        return ()

    @impl(Call)
    def run_call(self, interpreter: Interpreter, op: Call, args: tuple[Any, ...]):
        callee_op = interpreter.fetch_symbol(op.callee)
        if callee_op is None:
            raise InterpretationError(f"Didn't find @{op.callee.string_value()}")
        if not isinstance(callee_op, FuncOp):
            raise InterpretationError(f"Expected func.call to call a func.func.")

        ret = interpreter.run_ssacfg_region(callee_op.body, args)
        return ret

    @impl(Return)
    def run_return(self, interpreter: Interpreter, op: Return, args: tuple[Any, ...]):
        return (None, args)
