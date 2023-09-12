from abc import ABC
from dataclasses import dataclass

from xdsl.dialects import builtin, riscv, riscv_scf, snitch, snitch_stream
from xdsl.dialects.builtin import (
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


class SnitchStreamStreamToSnitch(RewritePattern, ABC):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch_stream.Stream, rewriter: PatternRewriter):
        dim = IntegerAttr(op.dimension.value, i32)

        all_streams = riscv.LiOp(snitch.SnitchResources.streams, comment="all streams")
        bound = riscv.LiOp(
            op.bound, comment=f"stream bound for dimension {op.dimension.value.data}"
        )
        stride = riscv.LiOp(
            op.stride, comment=f"stream stride for dimension {op.dimension.value.data}"
        )

        ssr_bound_op = snitch.SsrSetDimensionBoundOp(all_streams, bound, dim)
        ssr_stride_op = snitch.SsrSetDimensionStrideOp(all_streams, stride, dim)

        ops: list[Operation] = [all_streams, bound, stride, ssr_bound_op, ssr_stride_op]

        new_regs: list[Operation] = []

        for idx, inp in enumerate(op.inputs):
            stream = riscv.LiOp(idx, comment=f"input stream {idx}")
            stream_reg = builtin.UnrealizedConversionCastOp.get(
                (inp,), (riscv.IntRegisterType.unallocated(),)
            )
            new_regs.append(stream_reg)
            ops.append(stream)
            ops.append(stream_reg)
            ops.append(snitch.SsrSetDimensionSourceOp(stream, stream_reg, dim))

        for idx, outp in enumerate(op.outputs):
            stream = riscv.LiOp(
                idx + len(op.inputs), comment=f"output stream {idx + len(op.inputs)}"
            )
            stream_reg = builtin.UnrealizedConversionCastOp.get(
                (outp,), (riscv.IntRegisterType.unallocated(),)
            )
            new_regs.append(stream_reg)
            ops.append(stream)
            ops.append(stream_reg)
            ops.append(snitch.SsrSetDimensionDestinationOp(stream, stream_reg, dim))

        assert len(op.outputs) == 1, "SnitchStream must have one result"

        lb = riscv.LiOp(0)
        step = riscv.LiOp(1)
        ops.append(lb)
        ops.append(step)

        for_op = riscv_scf.ForOp(
            lb,
            bound,
            step,
            [],
            rewriter.move_region_contents_to_new_regions(op.body),
        )

        for i, arg in enumerate(for_op.body.block.args):
            arg.replace_by(new_regs[i].results[0])
            for_op.body.block.erase_arg(arg)

        for_op.body.block.insert_arg(riscv.IntRegisterType.unallocated(), 0)

        ops.append(snitch.SsrEnable())
        ops.append(for_op)
        ops.append(snitch.SsrDisable())

        rewriter.replace_matched_op(ops)


class LowerSnitchStreamYield(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch_stream.Yield, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(riscv_scf.YieldOp())


@dataclass
class LowerSnitchStreamToSnitchPass(ModulePass):
    """ """

    name = "lower-snitchstream-to-snitch"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker1 = PatternRewriteWalker(
            SnitchStreamStreamToSnitch(), apply_recursively=False
        )
        walker2 = PatternRewriteWalker(
            LowerSnitchStreamYield(), apply_recursively=False
        )
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
