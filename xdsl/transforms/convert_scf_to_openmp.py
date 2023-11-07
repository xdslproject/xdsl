from typing import cast

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import builtin, memref, omp, scf
from xdsl.ir import Block, MLContext, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertYield(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.Yield, rewriter: PatternRewriter, /):
        if len(op.operands) > 0:
            # TODO
            return
        rewriter.replace_matched_op(omp.YieldOp(*op.operands))


class ConvertParallel(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ParallelOp, rewriter: PatternRewriter, /):
        if len(op.initVals) > 0:
            # TODO
            return
        parallel = omp.ParallelOp(
            regions=[Region(Block())],
            operands=[[], [], [], [], []],
        )
        with ImplicitBuilder(parallel.region):
            wsloop = omp.WsLoopOp(
                operands=[
                    op.lowerBound,
                    op.upperBound,
                    op.step,
                    [],
                    [],
                    [],
                    [],
                ],
                regions=[Region(Block())],
            )
            with ImplicitBuilder(wsloop.body):
                alloca_scope = memref.AllocaScopeOp(
                    result_types=[[]], regions=[op.detach_region(op.body)]
                )

            omp.TerminatorOp()
        scope_block = alloca_scope.scope.block
        loop_block = wsloop.body.block
        terminator = cast(Operation, scope_block.ops.last)

        scope_block.insert_op_before(
            memref.AllocaScopeReturnOp(operands=[[]]),
            cast(Operation, scope_block.last_op),
        )
        loop_block.insert_op_after(
            terminator.detach(), cast(Operation, loop_block.ops.last)
        )
        for arg in reversed(scope_block.args):
            newarg = wsloop.body.block.insert_arg(arg.type, 0)
            arg.replace_by(newarg)
            scope_block.erase_arg(arg)
        rewriter.replace_matched_op(parallel)


class ConvertScfToOpenMPPass(ModulePass):
    name = "convert-scf-to-openpm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertYield(), ConvertParallel()])
        ).rewrite_module(op)
        print(op)
