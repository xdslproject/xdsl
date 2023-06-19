from traitlets import Any
from xdsl.dialects.func import Call, FuncOp, Return
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError


@register_impls
class FuncFunctions(InterpreterFunctions):
    @impl(FuncOp)
    def run_func(self, interpreter: Interpreter, op: FuncOp, args: tuple[Any, ...]):
        """
        This is the interpretation of a func.func, so what you should do when you are
        interpreting a block containing a function definition.

        By default, it is ignored, because nothing specific is done to the function,
        and the rationale is that it's the call's responsability to find the func.func
        and interpret its body.

        This rationale allows to flexibly do otherwise, for example: interpreting a function
        declaration is compiling it, and interpreting the call is finding its
        compilation, and calling it.

        This could be somehow included in the interpreter's infrastructe, with the right traits;
        Have a generic hook for SymbolOps, returning a corresponding Callable. Then the corresponding
        call Op could just fetch the Callable matching the called Symbol, and call it with its operands'
        values.
        """
        return ()

    @impl(Call)
    def run_call(self, interpreter: Interpreter, op: Call, args: tuple[Any, ...]):
        """
        This interpret a function call, by simply interpreting its body with the call's
        operands values. It then return the returned values as its results values.
        """
        callee_op = interpreter.fetch_symbol(op.callee)
        if callee_op is None:
            raise InterpretationError(f"Didn't find @{op.callee.string_value()}")
        if not isinstance(callee_op, FuncOp):
            raise InterpretationError(f"Expected func.call to call a func.func.")

        ret = interpreter.run_ssacfg_region(callee_op.body, args)
        return ret

    @impl(Return)
    def run_return(self, interpreter: Interpreter, op: Return, args: tuple[Any, ...]):
        """
        func.return just return its operands' values, and return no successor, as to get out of the region.
        The corresponding func.call will forward those to its results values.
        """
        return (None, args)
