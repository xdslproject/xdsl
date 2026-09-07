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
        loop_args = args

        while True:
            # while "before" region is terminated by an scf.condition op, of which `args[0]` is the i1 condition operand and
            # `args[1:]` are the values forwarded to the after-region (if the condition holds) or returned as the results of the
            # op (otherwise).
            condition, *forwarded_args = interpreter.run_ssacfg_region(
                op.before_region, loop_args, "while.before_region"
            )
            if not condition:
                return tuple(forwarded_args)

            loop_args = interpreter.run_ssacfg_region(
                op.after_region, tuple(forwarded_args), "while.after_region"
            )

    @impl(scf.IndexSwitchOp)
    def run_index_switch(
        self, interpreter: Interpreter, op: scf.IndexSwitchOp, args: PythonValues
    ) -> PythonValues:
        case_to_region_map = dict(
            zip(op.cases.iter_values(), op.case_regions, strict=True)
        )
        index = args[0]
        default = False
        if index in case_to_region_map:
            region = case_to_region_map[index]
        else:
            default = True
            region = op.default_region
        yielded_results = interpreter.run_ssacfg_region(
            region, (), f"index_switch.case {'default' if default else index}"
        )
        return yielded_results
