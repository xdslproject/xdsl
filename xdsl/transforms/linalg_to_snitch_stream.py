from abc import ABC
from dataclasses import dataclass

from xdsl.dialects import linalg, snitch_stream
from xdsl.dialects.builtin import (
    Float32Type,
    IntegerAttr,
    ModuleOp,
    i32,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerLinalgGenericToSnitchStreamStream(RewritePattern, ABC):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        loop_ranges = op.get_static_loop_ranges()

        bitwidth = 32
        if isinstance(op.body.block.args[0].type, Float32Type):
            bitwidth = 32

        for loop_idx in range(0, op.get_num_loops()):
            dim = IntegerAttr(loop_idx, i32)
            bound = IntegerAttr(loop_ranges[loop_idx], i32)

            stride = IntegerAttr(bitwidth // 8, i32)

            ss = snitch_stream.Stream(
                op.inputs, op.outputs, op.body.clone(), dim, bound, stride
            )

            rewriter.replace_matched_op(ss)


class LowerLinalgYieldToSnitchStreamYield(RewritePattern, ABC):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Yield, rewriter: PatternRewriter):
        rewriter.replace_matched_op(snitch_stream.Yield(*op.values))


@dataclass
class LowerLinalgToSnitchStreamPass(ModulePass):
    """ """

    name = "lower-linalg-to-snitch-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker1 = PatternRewriteWalker(
            LowerLinalgGenericToSnitchStreamStream(), apply_recursively=False
        )
        walker2 = PatternRewriteWalker(
            LowerLinalgYieldToSnitchStreamYield(), apply_recursively=False
        )
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
