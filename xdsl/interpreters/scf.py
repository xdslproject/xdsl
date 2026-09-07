from typing import Any

from xdsl.dialects import scf
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
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
        lb, ub, step, *packed_args = args
        loop_args = tuple(packed_args)

        for i in range(lb, ub, step):
            loop_args = interpreter.run_ssacfg_region(
                op.body, (i, *loop_args), "for_loop"
            )

        return loop_args

    @impl_terminator(scf.YieldOp)
    def run_br(
        self, interpreter: Interpreter, op: scf.YieldOp, args: tuple[Any, ...]
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()

    @impl_terminator(scf.ConditionOp)
    def run_condition(
        self, interpreter: Interpreter, op: scf.ConditionOp, args: PythonValues
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()

    @impl(scf.WhileOp)
    def run_while(
        self, interpreter: Interpreter, op: scf.WhileOp, args: PythonValues
    ) -> PythonValues:
        results = args

        while True:
            before_region = op.before_region
            results = interpreter.run_ssacfg_region(
                before_region, (*results,), "while.before_region"
            )
            if not results[0]:
                break

            results = results[1:]
            after_region = op.after_region
            results = interpreter.run_ssacfg_region(
                after_region, (*results,), "while.after_region"
            )
        return results[1:]

    def run_index_switch(
        self, interpreter: Interpreter, op: scf.IndexSwitchOp, args: PythonValues
    ) -> PythonValues:
        return ()
