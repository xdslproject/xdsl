from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, csl, memref
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    Signedness,
    StridedLayoutAttr,
)
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class LowerAllocOpPass(RewritePattern):
    """Lowers `memref.alloc` to `csl.zeros`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter, /):
        assert isa(op.memref.type, MemRefType[csl.ZerosOp.T])
        zeros_op = csl.ZerosOp(op.memref.type)
        # dsd_op = csl.GetMemDsdOp.from_memref(op.memref.type)

        dsd_t = csl.DsdType(
            csl.DsdKind.mem1d_dsd
            if len(op.memref.type.shape) == 1
            else csl.DsdKind.mem4d_dsd
        )
        offsets = None
        if isinstance(op.memref.type.layout, StridedLayoutAttr) and isinstance(
            op.memref.type.layout.offset, IntAttr
        ):
            offsets = ArrayAttr([IntegerAttr(op.memref.type.layout.offset, 16)])

        dsd_op = csl.GetMemDsdOp.build(
            operands=[zeros_op, []],
            result_types=[dsd_t],
            properties={
                "sizes": ArrayAttr(IntegerAttr(d, 16) for d in op.memref.type.shape),
                "offsets": offsets,
            },
        )

        rewriter.replace_matched_op([zeros_op, dsd_op])


class LowerSubviewOpPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Subview, rewriter: PatternRewriter, /):
        new_ops: list[Operation] = []

        curr_op = op
        assert len(op.static_sizes.data) == 1, "not implemented"
        assert len(op.static_offsets.data) == 1, "not implemented"
        assert len(op.static_strides.data) == 1, "not implemented"
        if op.static_sizes.data.data[0].data == memref.Subview.DYNAMIC_INDEX:
            pass
        else:
            new_ops.append(
                len_op := arith.Constant(op.static_sizes.data.data[0], csl.u16_value)
            )
            new_ops.append(
                curr_op := csl.SetDsdLengthOp.build(
                    operands=[curr_op, len_op], result_types=[op.source.type]
                )
            )
        if op.static_strides.data.data[0].data == memref.Subview.DYNAMIC_INDEX:
            pass
        else:
            new_ops.append(
                stride_op := arith.Constant(
                    op.static_strides.data.data[0], IntegerType(8, Signedness.SIGNED)
                )
            )
            new_ops.append(
                curr_op := csl.SetDsdStrideOp.build(
                    operands=[curr_op, stride_op], result_types=[op.source.type]
                )
            )
        if op.static_offsets.data.data[0].data == memref.Subview.DYNAMIC_INDEX:
            pass
        else:
            pass

        rewriter.replace_matched_op(new_ops)


@dataclass(frozen=True)
class MemrefToDsdPass(ModulePass):
    """
    Lowers memref ops to CSL DSDs.

    Note, that CSL uses memref types in some places
    """

    name = "memref-to-dsd"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAllocOpPass(),
                    LowerSubviewOpPass(),
                ]
            )
        )
        module_pass.rewrite_module(op)
