from typing import Any
from xdsl.dialects import scf

from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)


@register_impls
class ScfFunctions(InterpreterFunctions):
    @impl(scf.For)
    def run_for(
        self, interpreter: Interpreter, op: scf.For, args: PythonValues
    ) -> PythonValues:
        for i in range(*args):
            loop_results = interpreter.run_ssacfg_region(op.body, (i,), "for_loop")
            assert not loop_results

        return ()

    @impl_terminator(scf.Yield)
    def run_br(self, interpreter: Interpreter, op: scf.Yield, args: tuple[Any, ...]):
        assert not args
        return ReturnedValues(()), ()
