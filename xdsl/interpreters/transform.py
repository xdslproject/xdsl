from collections.abc import Callable
from typing import TypeAlias

from xdsl.context import Context
from xdsl.dialects import builtin, transform
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
from xdsl.dialects.linalg.transforms.tiling import tile_structured_op
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
from xdsl.pattern_rewriter import PatternRewriter
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

    @impl(transform.MatchOp)
    def run_match_op(
        self,
        interpreter: Interpreter,
        op: transform.MatchOp,
        args: PythonValues,
    ) -> PythonValues:
        (roots,) = args
        assert isa(roots, OperationHandle)
        if len(roots) != 1:
            raise InterpretationError(
                "transform.structured.match requires exactly one target operation"
            )

        for name, value in (
            ("interface", op.interface),
            ("op_attrs", op.op_attrs),
            ("filter_result_type", op.filter_result_type),
            ("filter_operand_types", op.filter_operand_types),
        ):
            if value is not None:
                raise InterpretationError(
                    f"transform.structured.match does not yet support the {name} filter"
                )

        names = None if op.ops is None else {name.data for name in op.ops}
        # MLIR walks the root and its descendants in post-order. An absent name
        # filter matches everything, whereas an explicitly empty one matches nothing.
        matches = tuple(
            candidate
            for candidate in roots[0].walk(region_first=True)
            if names is None or candidate.name in names
        )
        return (matches,)

    @impl(transform.TileOp)
    def run_tile_op(
        self,
        interpreter: Interpreter,
        op: transform.TileOp,
        args: PythonValues,
    ) -> PythonValues:
        targets = args[0]
        assert isa(targets, OperationHandle)
        if op.dynamic_sizes:
            raise InterpretationError(
                "transform.structured.tile_using_for does not yet support dynamic tile sizes"
            )
        if op.scalable_sizes is not None and any(op.scalable_sizes.get_values()):
            raise InterpretationError(
                "transform.structured.tile_using_for does not yet support scalable tile sizes"
            )
        if op.interchange is not None and len(op.interchange):
            raise InterpretationError(
                "transform.structured.tile_using_for does not yet support interchange"
            )
        sizes = op.static_sizes.get_values() if op.static_sizes is not None else ()
        if any(size < 0 for size in sizes):
            raise InterpretationError(
                "transform.structured.tile_using_for requires nonnegative tile sizes"
            )
        if not isa(targets, tuple[LinalgStructuredOperation, ...]):
            raise InterpretationError(
                "transform.structured.tile_using_for supports only structured linalg targets"
            )
        for target in targets:
            num_loops = target.get_num_loops()
            if len(sizes) > num_loops:
                raise InterpretationError(
                    f"transform.structured.tile_using_for expected at most {num_loops} "
                    f"tile sizes for {target.name}, got {len(sizes)}"
                )

        tiled_ops: list[Operation] = []
        loops: list[list[Operation]] = [[] for _ in op.loops]
        for target in targets:
            rewriter = PatternRewriter(target)
            result = tile_structured_op(rewriter, target, sizes)
            tiled_ops.append(result.tiled_op)
            # Each transform result groups the corresponding loop across targets.
            for loop_handle, loop in zip(loops, result.loops, strict=True):
                loop_handle.append(loop)

        return (tuple(tiled_ops), *(tuple(loop_handle) for loop_handle in loops))

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
