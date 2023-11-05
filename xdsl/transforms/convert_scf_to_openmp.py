from xdsl.dialects import builtin, omp, scf
from xdsl.ir import Block, MLContext, Region
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
            regions=[
                Region(
                    Block(
                        [
                            omp.WsLoopOp(
                                operands=[
                                    op.lowerBound,
                                    op.upperBound,
                                    op.step,
                                    [],
                                    [],
                                    [],
                                    [],
                                ],
                                regions=[op.detach_region(op.body)],
                            ),
                            omp.TerminatorOp(),
                        ]
                    )
                )
            ],
            operands=[[], [], [], [], []],
        )
        rewriter.replace_matched_op(parallel)


class ConvertScfToOpenMPPass(ModulePass):
    name = "convert-scf-to-openpm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertYield(), ConvertParallel()])
        ).rewrite_module(op)
