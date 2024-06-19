from dataclasses import dataclass
from typing import Literal

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, memref, omp, scf
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class ConvertParallel(RewritePattern):
    collapse: int | None
    nested: bool
    schedule: Literal["static", "dynamic", "auto"] | None
    chunk: int | None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, loop: scf.ParallelOp, rewriter: PatternRewriter, /):
        if len(loop.initVals) > 0:
            # TODO Implement reduction, see https://github.com/xdslproject/xdsl/issues/1776
            return

        collapse = self.collapse
        if collapse is None or collapse > len(loop.lowerBound):
            collapse = len(loop.lowerBound)

        if not self.nested:
            parent = loop
            while (parent := parent.parent_op()) is not None:
                if isinstance(parent, omp.WsLoopOp):
                    return

        parallel = omp.ParallelOp(
            regions=[Region(Block())],
            operands=[[], [], [], [], []],
        )
        with ImplicitBuilder(parallel.region):
            if self.chunk is None:
                chunk_op = []
            else:
                self.schedule = "static"
                chunk_op = [arith.Constant.from_int_and_width(self.chunk, IndexType())]
            wsloop = omp.WsLoopOp(
                operands=[
                    loop.lowerBound[:collapse],
                    loop.upperBound[:collapse],
                    loop.step[:collapse],
                    [],
                    [],
                    [],
                    chunk_op,
                ],
                regions=[Region(Block(arg_types=[IndexType()] * collapse))],
            )
            if self.schedule is not None:
                wsloop.schedule_val = omp.ScheduleKindAttr(
                    omp.ScheduleKind(self.schedule)
                )
            omp.TerminatorOp()
        with ImplicitBuilder(wsloop.body):
            scope = memref.AllocaScopeOp(result_types=[[]], regions=[Region(Block())])
            omp.YieldOp()
        with ImplicitBuilder(scope.scope):
            scope_terminator = memref.AllocaScopeReturnOp(operands=[[]])
        for newarg, oldarg in zip(
            wsloop.body.block.args, loop.body.block.args[:collapse]
        ):
            oldarg.replace_by(newarg)
        for _ in range(collapse):
            loop.body.block.erase_arg(loop.body.block.args[0])
        if collapse < len(loop.lowerBound):
            new_loop = scf.ParallelOp(
                lower_bounds=loop.lowerBound[collapse:],
                upper_bounds=loop.upperBound[collapse:],
                steps=loop.step[collapse:],
                body=loop.detach_region(loop.body),
            )
            new_ops = [new_loop]
        else:
            new_ops = [loop.body.block.detach_op(o) for o in loop.body.block.ops]
            new_ops.pop()
        scope.scope.block.insert_ops_before(new_ops, scope_terminator)

        rewriter.replace_matched_op(parallel)


@dataclass(frozen=True)
class ConvertScfToOpenMPPass(ModulePass):
    """
    Convert `scf.parallel` loops to `omp.wsloop` constructs for parallel execution.
    It currently does not support reduction.

    Arguments (all optional):

    - collapse : int: specify a positive number of loops to collapse. By default, the
    full dimensionality of converted parallel loops is collapsed. This argument
    allows to take a 2D loop and only OMPize the first dimension, for example.

    - nested: bool: Set this to true to convert nested parallel loops. This is
    rarely a good idea, and is disabled by default. Note that setting it to true mimics
    MLIR's convert-scf-to-openmp.

    - schedule: {"static", "dynamic", "auto"}: Set the schedule used by the OMP loop.
    By default, none is set, leaving the decision to MLIR's omp lowering. At the time
    of writing, this means static.

    - chunk: int: Set the chunk size used by the OMP loop. By default, none is set.
    Note that the OMP dialect does not support setting a chunk size without a schedule;
    Thus selecting a chunk size without a schedule will use the static schedule.
    """

    name = "convert-scf-to-openmp"

    collapse: int | None = None
    nested: bool = False
    schedule: Literal["static", "dynamic", "auto"] | None = None
    chunk: int | None = None

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertParallel(
                        self.collapse, self.nested, self.schedule, self.chunk
                    ),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
