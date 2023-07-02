from dataclasses import dataclass

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import Block, MLContext, Region, Operation, OpResult
from xdsl.irdl import VarOperand, VarOpResult
from xdsl.dialects.func import FuncOp, Return
from xdsl.dialects.func import Call
from xdsl.dialects import builtin
from xdsl.dialects.builtin import i32, IndexType
from xdsl.dialects.arith import Constant

from xdsl.dialects.experimental.hls import PragmaPipeline, PragmaUnroll, PragmaDataflow

from xdsl.passes import ModulePass

from xdsl.dialects.scf import ParallelOp, For, Yield

from typing import cast, Any


@dataclass
class PragmaPipelineToFunc(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaPipeline, rewriter: PatternRewriter, /):
        # TODO: can we retrieve data directly without having to go through IntegerAttr -> IntAttr?
        # print("---->: ", op.ii.owner)
        # ii : i32 = op.ii.owner.value.value.data
        ii = cast(Any, op.ii.owner).value.value.data

        ret1 = Return()
        block1 = Block(arg_types=[])
        block1.add_ops([ret1])
        region1 = Region(block1)
        func1 = FuncOp.from_region(f"_pipeline_{ii}_", [], [], region1)

        call1 = Call.get(func1.sym_name.data, [], [])

        self.module.body.block.add_op(func1)

        rewriter.replace_matched_op(call1)


@dataclass
class PragmaUnrollToFunc(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaUnroll, rewriter: PatternRewriter, /):
        # TODO: can we retrieve data directly without having to go through IntegerAttr -> IntAttr?
        factor = cast(Any, op.factor.owner).value.value.data

        ret1 = Return()
        block1 = Block(arg_types=[])
        block1.add_ops([ret1])
        region1 = Region(block1)
        func1 = FuncOp.from_region(f"_unroll_{factor}_", [], [], region1)

        call1 = Call.get(func1.sym_name.data, [], [])

        self.module.body.block.add_op(func1)

        rewriter.replace_matched_op(call1)


@dataclass
class PragmaDataflowToFunc(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaDataflow, rewriter: PatternRewriter, /):
        # TODO: can we retrieve data directly without having to go through IntegerAttr -> IntAttr?
        ret1 = Return()
        block1 = Block(arg_types=[])
        block1.add_ops([ret1])
        region1 = Region(block1)
        func1 = FuncOp.from_region(f"_dataflow", [], [], region1)

        call1 = Call.get(func1.sym_name.data, [], [])

        self.module.body.block.add_op(func1)

        rewriter.replace_matched_op(call1)


class SCFParallelToHLSPipelinedFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ParallelOp, rewriter: PatternRewriter, /):
        ii = Constant.from_int_and_width(1, i32)
        hls_pipeline_op: Operation = PragmaPipeline.get(ii)

        lb: VarOperand = op.lowerBound
        ub: VarOperand = op.upperBound
        step: VarOperand = op.step
        res: VarOpResult = op.res

        for i in range(len(lb)):
            # print(lb[i])
            cast(OpResult, lb[i]).op.detach()
            cast(OpResult, ub[i]).op.detach()
            cast(OpResult, step[i]).op.detach()

        # We generate a For loop for each induction variable in the Parallel loop.
        # We start by wrapping the parallel block in a region for the For loop and keep
        # wrapping in for loops until we have exhausted the induction variables
        parallel_block = op.body.detach_block(0)

        if res != []:
            parallel_block.insert_arg(res[0].typ, 1)
            cast(Operation, parallel_block.last_op).detach()
            yieldop = Yield.get(res[0].op)
            parallel_block.add_op(yieldop)

        for_region = Region([parallel_block])

        for i in range(len(lb) - 1):
            for_region.block.erase_arg(for_region.block.args[i])

        if res != []:
            for_op = For.get(lb[-1], ub[-1], step[-1], [res[0].op], for_region)
        else:
            for_op = For.get(lb[-1], ub[-1], step[-1], [], for_region)

        for i in range(len(lb) - 2, -1, -1):
            for_region = Region(Block([for_op]))

            for_region.block.insert_arg(IndexType(), 0)

            for_region.block.insert_op_before(
                cast(OpResult, lb[i + 1]).op, cast(Operation, for_region.block.first_op)
            )
            for_region.block.insert_op_after(
                cast(OpResult, ub[i + 1]).op, cast(OpResult, lb[i + 1]).op
            )
            for_region.block.insert_op_after(
                cast(OpResult, step[i + 1]).op, cast(OpResult, ub[i + 1]).op
            )
            yieldop = Yield.get()
            for_region.block.add_op(yieldop)
            for_op = For.get(lb[i], ub[i], step[i], [], for_region)

        for_region.block.insert_op_before(
            hls_pipeline_op, cast(Operation, for_region.block.first_op)
        )
        for_region.block.insert_op_after(ii, cast(Operation, for_region.block.first_op))

        cast(Block, op.parent_block()).insert_op_before(
            cast(OpResult, lb[0]).op,
            cast(Operation, cast(Block, op.parent_block()).first_op),
        )
        cast(Block, op.parent_block()).insert_op_after(
            cast(OpResult, ub[0]).op, cast(OpResult, lb[0]).op
        )
        cast(Block, op.parent_block()).insert_op_after(
            cast(OpResult, step[0]).op, cast(OpResult, ub[0]).op
        )

        rewriter.replace_matched_op([for_op])


@dataclass
class LowerHLSPass(ModulePass):
    name = "lower-hls"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        def gen_greedy_walkers(
            passes: list[RewritePattern],
        ) -> list[PatternRewriteWalker]:
            # Creates a greedy walker for each pass, so that they can be run sequentially even after
            # matching
            walkers: list[PatternRewriteWalker] = []

            for i in range(len(passes)):
                walkers.append(
                    PatternRewriteWalker(
                        GreedyRewritePatternApplier([passes[i]]), apply_recursively=True
                    )
                )

            return walkers

        walkers = gen_greedy_walkers(
            [
                SCFParallelToHLSPipelinedFor(),
                PragmaPipelineToFunc(op),
                PragmaUnrollToFunc(op),
                PragmaDataflowToFunc(op),
            ]
        )

        for walker in walkers:
            walker.rewrite_module(op)
