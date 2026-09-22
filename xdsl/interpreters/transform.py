from collections.abc import Callable
from typing import TypeAlias

from xdsl.context import Context
from xdsl.dialects import builtin, transform
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
from xdsl.ir import Operation
from xdsl.passes import ModulePass, PassPipeline
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa

OperationHandle: TypeAlias = tuple[Operation, ...]
"""
The payload operations associated with one transform handle. Like MLIR's
TransformState mapping, this preserves storage order, but that order has no
semantic meaning unless the transform operation specifies otherwise.
"""


@register_impls
class TransformFunctions(InterpreterFunctions):
    """
    Interpret transform operations with each operation handle represented by an
    `OperationHandle`, including empty and single-operation handles. Each handle
    occupies one element of the interpreter's argument or result tuple.
    """

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
        (targets,) = args
        assert isa(targets, OperationHandle)
        if not isa(targets, tuple[builtin.ModuleOp, ...]):
            raise InterpretationError(
                "transform.apply_registered_pass currently supports only builtin.module targets"
            )
        pass_name = op.pass_name.data
        pipeline = PassPipeline.parse_spec(self.passes, pass_name)
        for target in targets:
            pipeline.apply(self.ctx, target)
        return (targets,)

    @impl_terminator(transform.YieldOp)
    def run_yield_op(
        self, interpreter: Interpreter, op: transform.YieldOp, args: PythonValues
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()
