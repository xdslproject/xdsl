from dataclasses import dataclass
from typing import Literal

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
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
from xdsl.rewriter import InsertPoint


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
            operands=[[], [], [], [], [], []],
        )
        rewriter.insertion_point = InsertPoint.at_end(parallel.region.block)
        with ImplicitBuilder(rewriter):
            if self.chunk is None:
                chunk_op = []
            else:
                self.schedule = "static"
                chunk_op = [
                    arith.ConstantOp.from_int_and_width(self.chunk, IndexType())
                ]
            wsloop = omp.WsLoopOp(
                operands=[
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    chunk_op,
                ],
                regions=[Region(Block())],
            )
            if self.schedule is not None:
                wsloop.schedule_kind = omp.ScheduleKindAttr(
                    omp.ScheduleKind(self.schedule)
                )
            omp.TerminatorOp()

        rewriter.insertion_point = InsertPoint.at_end(wsloop.body.block)
        with ImplicitBuilder(rewriter):
            loop_nest = omp.LoopNestOp(
                operands=[
                    loop.lowerBound[:collapse],
                    loop.upperBound[:collapse],
                    loop.step[:collapse],
                ],
                regions=[Region(Block(arg_types=[IndexType()] * collapse))],
            )

        rewriter.insertion_point = InsertPoint.at_end(loop_nest.body.block)
        with ImplicitBuilder(rewriter):
            scope = memref.AllocaScopeOp(result_types=[[]], regions=[Region(Block())])
            omp.YieldOp()

        rewriter.insertion_point = InsertPoint.at_end(scope.scope.block)
        with ImplicitBuilder(rewriter):
            scope_terminator = memref.AllocaScopeReturnOp(operands=[[]])

        for newarg, oldarg in zip(
            loop_nest.body.block.args, loop.body.block.args[:collapse]
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
            last_op = new_ops.pop()
            rewriter.erase_op(last_op)
        rewriter.insert_op(new_ops, InsertPoint.before(scope_terminator))

        rewriter.replace_op(loop, parallel)


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

    def apply(self, ctx: Context, op: ModuleOp) -> None:
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
