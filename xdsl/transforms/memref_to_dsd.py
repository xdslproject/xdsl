from dataclasses import dataclass
from typing import cast

from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, csl, memref
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float16Type,
    Float32Type,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    NoneAttr,
    Signedness,
    StridedLayoutAttr,
)
from xdsl.ir import Attribute, Operation, SSAValue
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

        shape = [arith.Constant(IntegerAttr(d, 16)) for d in op.memref.type.shape]
        dsd_op = csl.GetMemDsdOp.build(
            operands=[zeros_op, shape],
            result_types=[dsd_t],
            properties={
                "offsets": offsets,
            },
        )

        rewriter.replace_matched_op([zeros_op, *shape, dsd_op])


class LowerSubviewOpPass(RewritePattern):
    """Lowers memref.subview to dsd ops"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Subview, rewriter: PatternRewriter, /):
        assert isa(op.source.type, MemRefType[Attribute])
        assert len(op.static_sizes.data) == 1, "not implemented"
        assert len(op.static_offsets.data) == 1, "not implemented"
        assert len(op.static_strides.data) == 1, "not implemented"

        last_op = op.source
        size_ops = self._update_sizes(op, last_op)

        last_op = size_ops[-1] if len(size_ops) > 0 else last_op
        stride_ops = self._update_strides(op, last_op)

        last_op = stride_ops[-1] if len(stride_ops) > 0 else last_op
        offset_ops = self._update_offsets(op, last_op)

        rewriter.replace_matched_op([*size_ops, *stride_ops, *offset_ops])

    @staticmethod
    def _update_sizes(
        subview: memref.Subview, curr_op: SSAValue | Operation
    ) -> list[Operation]:
        assert isa(subview.source.type, MemRefType[Attribute])
        ops = list[Operation]()

        if subview.static_sizes.data.data[0].data == memref.Subview.DYNAMIC_INDEX:
            ops.append(cast_op := arith.IndexCastOp(subview.sizes[0], csl.u16_value))
            ops.append(
                curr_op := csl.SetDsdLengthOp.build(
                    operands=[curr_op, cast_op], result_types=[subview.source.type]
                )
            )
        elif subview.static_sizes.as_tuple() != subview.source.type.get_shape():
            # update sizes only if they differ from op.source.type
            ops.append(
                len_op := arith.Constant(
                    IntegerAttr(
                        cast(ArrayAttr[IntAttr], subview.static_sizes.data).data[0],
                        csl.u16_value,
                    )
                )
            )
            ops.append(
                curr_op := csl.SetDsdLengthOp.build(
                    operands=[curr_op, len_op], result_types=[subview.source.type]
                )
            )
        return ops

    @staticmethod
    def _update_strides(
        subview: memref.Subview, curr_op: SSAValue | Operation
    ) -> list[Operation]:
        assert isa(subview.source.type, MemRefType[Attribute])
        ops = list[Operation]()

        if subview.static_strides.data.data[0].data == memref.Subview.DYNAMIC_INDEX:
            ops.append(
                cast_op := arith.IndexCastOp(
                    subview.strides[0], IntegerType(8, Signedness.SIGNED)
                )
            )
            ops.append(
                csl.SetDsdStrideOp.build(
                    operands=[curr_op, cast_op], result_types=[subview.source.type]
                )
            )
        elif subview.static_strides.as_tuple() != subview.source.type.get_strides():
            # update strides only if they differ from op.source.type
            ops.append(
                stride_op := arith.Constant(
                    IntegerAttr(
                        cast(ArrayAttr[IntAttr], subview.static_strides.data).data[0],
                        IntegerType(8, Signedness.SIGNED),
                    )
                )
            )
            ops.append(
                csl.SetDsdStrideOp.build(
                    operands=[curr_op, stride_op], result_types=[subview.source.type]
                )
            )
        return ops

    @staticmethod
    def _update_offsets(
        subview: memref.Subview, curr_op: SSAValue | Operation
    ) -> list[Operation]:
        assert isa(subview.source.type, MemRefType[Attribute])
        ops = list[Operation]()

        if subview.static_offsets.data.data[0].data == memref.Subview.DYNAMIC_INDEX:
            ops.append(cast_op := arith.IndexCastOp(subview.offsets[0], csl.i16_value))
            ops.append(
                csl.IncrementDsdOffsetOp.build(
                    operands=[curr_op, cast_op],
                    properties={"elem_type": subview.source.type.get_element_type()},
                    result_types=[subview.source.type],
                )
            )
        elif (
            isinstance(subview.source.type.layout, StridedLayoutAttr)
            and subview.static_offsets.as_tuple()[0]
            != (subview.source.type.layout.get_offset() or 0)
            or isinstance(subview.source.type.layout, NoneAttr)
            and subview.static_offsets.as_tuple()[0] != 0
        ):
            # update offsets only if they differ from op.source.type
            ops.append(
                offset_op := arith.Constant(
                    IntegerAttr(
                        cast(ArrayAttr[IntAttr], subview.static_offsets.data).data[0],
                        csl.i16_value,
                    )
                )
            )
            ops.append(
                csl.IncrementDsdOffsetOp.build(
                    operands=[curr_op, offset_op],
                    properties={"elem_type": subview.source.type.get_element_type()},
                    result_types=[subview.source.type],
                )
            )
        return ops


class LowerCopyOpPass(RewritePattern):
    """Lowers memref.copy to csl"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CopyOp, rewriter: PatternRewriter, /):
        assert isa(op.source.type, MemRefType[Attribute])

        match op.source.type.get_element_type():
            case Float16Type():
                func = csl.FmovhOp
            case Float32Type():
                func = csl.FmovsOp
            case builtin.i16:
                func = csl.Mov16Op
            case builtin.i32:
                func = csl.Mov32Op
            case _:
                raise ValueError("unsupported value")

        rewriter.replace_matched_op(func(operands=[[op.destination, op.source]]))


class DsdOpUpdateType(RewritePattern):
    """Rebuild DSD ops from memref to DSD types."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: csl.IncrementDsdOffsetOp | csl.SetDsdStrideOp | csl.SetDsdLengthOp,
        rewriter: PatternRewriter,
        /,
    ):
        rewriter.replace_matched_op(
            type(op).build(
                operands=op.operands,
                properties=op.properties,
                attributes=op.attributes,
                result_types=[op.op.type],
            )
        )


class RetainAddressOfOpPass(RewritePattern):
    """Ensure we don't export DSD but the underlying memref."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.AddressOfOp, rewriter: PatternRewriter, /):
        if isinstance(op.value.type, csl.DsdType) and isinstance(
            op.value.owner, csl.GetMemDsdOp
        ):
            rewriter.replace_matched_op(
                csl.AddressOfOp.build(
                    operands=[op.value.owner.base_addr], result_types=op.result_types
                )
            )


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
                    LowerSubviewOpPass(),
                    LowerCopyOpPass(),
                ]
            ),
            walk_reverse=True,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
        forward_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAllocOpPass(),
                    DsdOpUpdateType(),
                    RetainAddressOfOpPass(),
                ]
            ),
            apply_recursively=False,
        )
        forward_pass.rewrite_module(op)
