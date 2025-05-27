from collections.abc import Callable

from xdsl.context import Context
from xdsl.dialects import transform
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl,
    impl_callable,
    impl_terminator,
    register_impls,
)
from xdsl.passes import ModulePass, PipelinePass
from xdsl.utils import parse_pipeline


@register_impls
class TransformFunctions(InterpreterFunctions):
    ctx: Context
    passes: dict[str, Callable[[], type[ModulePass]]]

    def __init__(
        self, ctx: Context, available_passes: dict[str, Callable[[], type[ModulePass]]]
    ):
        self.ctx = ctx
        self.passes = available_passes

    @impl_callable(transform.NamedSequenceOp)
    def run_named_sequence_op(
        self,
        interpreter: Interpreter,
        op: transform.NamedSequenceOp,
        args: PythonValues,
    ) -> PythonValues:
        return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)

    @impl(transform.ApplyRegisteredPassOp)
    def run_apply_registered_pass_op(
        self,
        interpreter: Interpreter,
        op: transform.ApplyRegisteredPassOp,
        args: PythonValues,
    ) -> PythonValues:
        pass_name = op.pass_name.data

        schedule = tuple(
            PipelinePass.iter_passes(
                self.passes, parse_pipeline.parse_pipeline(pass_name)
            )
        )
        pipeline = PipelinePass(schedule)
        pipeline.apply(self.ctx, args[0])
        return (args[0],)

    @impl_terminator(transform.YieldOp)
    def run_yield_op(
        self, interpreter: Interpreter, op: transform.YieldOp, args: PythonValues
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()
