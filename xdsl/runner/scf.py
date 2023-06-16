from traitlets import Any
from xdsl.dialects.scf import If, Yield
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls


@register_impls
class ScfFunctions(InterpreterFunctions):
    @impl(If)
    def run_if(self, interpreter: Interpreter, op: If, args: tuple[Any, ...]):
        return interpreter.run_ssacfg_region(
            op.true_region if args[0] else op.false_region, args[1:]
        )

    @impl(Yield)
    def run_yield(self, interpreter: Interpreter, op: Yield, args: tuple[Any, ...]):
        return None, tuple(args)
