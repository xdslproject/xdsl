from dataclasses import dataclass

from xdsl.builder import InsertPoint
from xdsl.context import MLContext
from xdsl.dialects import builtin, func
from xdsl.dialects.experimental.hida_structural import NodeOp, ScheduleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class InlineSchedule(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, schedule: ScheduleOp, rewriter: PatternRewriter):
        for t in zip(schedule.region.block.args, schedule.operands):
            t[0].replace_by(t[1])

        rewriter.inline_block(schedule.region.block, InsertPoint.before(schedule))

        rewriter.erase_matched_op()


@dataclass
class ConvertNodeToFunc(RewritePattern):
    module: builtin.ModuleOp
    node_idx = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, node: NodeOp, rewriter: PatternRewriter):
        sub_func_type = builtin.FunctionType.from_lists(
            [operand.type for operand in node.operands], []
        )
        sub_func = func.FuncOp(f"node_{self.__class__.node_idx}", sub_func_type)
        rewriter.insert_op(sub_func, InsertPoint.at_start(self.module.body.block))

        # TODO: inline attribute

        # Inline the contents of the dataflow node.
        rewriter.inline_region_before(node.region, sub_func.body.block)
        rewriter.insert_op(func.Return(), InsertPoint.at_end(sub_func.body.blocks[0]))

        # Replace original with a function call.
        rewriter.replace_matched_op(
            func.Call(f"node_{self.__class__.node_idx}", node.operands, [])
        )
        self.__class__.node_idx += 1


@dataclass(frozen=True)
class ConvertDataflowToFunc(ModulePass):
    name = "hida-convert-dataflow-to-func"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        lower_affine_mem = PatternRewriteWalker(
            GreedyRewritePatternApplier([InlineSchedule(), ConvertNodeToFunc(op)])
        )
        lower_affine_mem.rewrite_module(op)
