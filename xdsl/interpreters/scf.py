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
    @impl(scf.IfOp)
    def run_if(self, interpreter: Interpreter, op: scf.IfOp, args: tuple[Any, ...]):
        (cond,) = args
        region = op.true_region if cond else op.false_region
        results = interpreter.run_ssacfg_region(region, ())
        return results

    @impl(scf.ForOp)
    def run_for(
        self, interpreter: Interpreter, op: scf.ForOp, args: PythonValues
    ) -> PythonValues:
        lb, ub, step, *loop_args = args
        loop_args = tuple(loop_args)

        for i in range(lb, ub, step):
            loop_args = interpreter.run_ssacfg_region(
                op.body, (i, *loop_args), "for_loop"
            )

        return loop_args

    @impl_terminator(scf.YieldOp)
    def run_br(self, interpreter: Interpreter, op: scf.YieldOp, args: tuple[Any, ...]):
        return ReturnedValues(args), ()
