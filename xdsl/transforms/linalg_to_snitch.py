from abc import ABC
from dataclasses import dataclass

from xdsl.dialects import linalg, riscv, snitch
from xdsl.dialects.builtin import (
    Float32Type,
    IntegerAttr,
    ModuleOp,
    i32,
)
from xdsl.ir import MLContext
from xdsl.ir.core import Operation  # noqa: E999
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddSnitchStreamLoopBoundAndStrideConfig(RewritePattern, ABC):
    """
    Adds Snitch stream configuration instructions for the loop bound and stride at a
    specific loop depth.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        for loop_idx in range(0, op.get_num_loops()):
            stream = riscv.LiOp(snitch.SnitchResources.streams)
            bound = riscv.LiOp(364)
            dim = IntegerAttr(loop_idx, i32)
            ssr_bound_op = snitch.SsrSetDimensionBoundOp(stream, bound, dim)

            stride = riscv.LiOp(4)
            if isinstance(op.body.block.args[0].type, Float32Type):
                stride = riscv.LiOp(32 // 8)

            ssr_stride_op = snitch.SsrSetDimensionStrideOp(stream, stride, dim)

            rewriter.insert_op_before_matched_op(
                [stream, bound, stride, ssr_bound_op, ssr_stride_op]
            )


class AddSnitchStreamSetSourceDestinationConfig(RewritePattern, ABC):
    """
    Adds Snitch stream configuration instructions for a source stream register.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        for loop_idx in range(0, op.get_num_loops()):
            dim = IntegerAttr(loop_idx, i32)

            ops: list[Operation] = []
            for idx, _ in enumerate(op.inputs):
                stream = riscv.LiOp(idx)
                stream_reg = riscv.GetRegisterOp(riscv.IntRegisterType.unallocated())
                ops.append(stream)
                ops.append(stream_reg)
                ops.append(snitch.SsrSetDimensionSourceOp(stream, stream_reg, dim))

            for idx, _ in enumerate(op.outputs):
                stream = riscv.LiOp(idx + len(op.inputs))
                stream_reg = riscv.GetRegisterOp(riscv.IntRegisterType.unallocated())
                ops.append(stream)
                ops.append(stream_reg)
                ops.append(snitch.SsrSetDimensionDestinationOp(stream, stream_reg, dim))

            rewriter.insert_op_before_matched_op(ops)


class RemoveLinalgGeneric(RewritePattern, ABC):
    """ """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class AddSnitchStreamEnableDisable(RewritePattern, ABC):
    """
    Adds Snitch stream enable and disable instructions.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        rewriter.insert_op_before_matched_op(snitch.SsrEnable())
        rewriter.insert_op_after_matched_op(snitch.SsrDisable())


@dataclass
class LowerLinalgToSnitchPass(ModulePass):
    """ """

    name = "lower-linalg-to-snitch"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker1 = PatternRewriteWalker(
            AddSnitchStreamLoopBoundAndStrideConfig(), apply_recursively=False
        )
        walker2 = PatternRewriteWalker(
            AddSnitchStreamSetSourceDestinationConfig(), apply_recursively=False
        )
        walker3 = PatternRewriteWalker(
            AddSnitchStreamEnableDisable(), apply_recursively=False
        )
        walker4 = PatternRewriteWalker(RemoveLinalgGeneric(), apply_recursively=False)
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
        walker3.rewrite_module(op)
        walker4.rewrite_module(op)
