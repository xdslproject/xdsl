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
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa
from xdsl.utils.isattr import isattr


class LowerAllocOpPass(RewritePattern):
    """Lowers `memref.alloc` to `csl.zeros`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter, /):
        assert isattr(
            memref_type := op.memref.type,
            MemRefType.constr(element_type=csl.ZerosOpAttrConstr),
        )
        zeros_op = csl.ZerosOp(memref_type)

        dsd_t = csl.DsdType(
            csl.DsdKind.mem1d_dsd
            if len(memref_type.shape) == 1
            else csl.DsdKind.mem4d_dsd
        )
        offsets = None
        if isinstance(memref_type.layout, StridedLayoutAttr) and isinstance(
            memref_type.layout.offset, IntAttr
        ):
            offsets = ArrayAttr([IntegerAttr(memref_type.layout.offset, 16)])

        shape = [arith.ConstantOp(IntegerAttr(d, 16)) for d in memref_type.shape]
        dsd_op = csl.GetMemDsdOp.build(
            operands=[zeros_op, shape],
            result_types=[dsd_t],
            properties={
                "offsets": offsets,
            },
        )

        if op.memref.name_hint:
            zeros_op.result.name_hint = op.memref.name_hint
            dsd_op.result.name_hint = f"{op.memref.name_hint}_dsd"
            for s in shape:
                s.result.name_hint = f"{op.memref.name_hint}_size"

        rewriter.replace_matched_op([zeros_op, *shape, dsd_op])


class FixGetDsdOnGetDsd(RewritePattern):
    """
    This rewrite pattern resolves GetMemDsdOp being called on GetMemDsdOp instead of the underlying buffer,
    a side effect created by `LowerAllocOpPass` in case of pre-existing GetMemDsdOp ops being present in
    the program that were created outside of this pass.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter, /):
        if isinstance(op.base_addr.type, csl.DsdType):
            if isinstance(op.base_addr, OpResult) and isinstance(
                op.base_addr.op, csl.GetMemDsdOp
            ):
                rewriter.replace_matched_op(
                    csl.GetMemDsdOp.build(
                        operands=[op.base_addr.op.base_addr, op.sizes],
                        properties=op.properties,
                        attributes=op.attributes,
                        result_types=op.result_types,
                    )
                )
            else:
                raise ValueError("Failed to resolve GetMemDsdOp called on dsd type")


class FixMemrefLoadOnGetDsd(RewritePattern):
    """
    Memref load ops should load from the underlying memref, not from the dsd.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        if isinstance(op.memref.type, csl.DsdType):
            if isinstance(op.memref, OpResult) and isinstance(
                op.memref.op, csl.GetMemDsdOp
            ):
                rewriter.replace_matched_op(
                    memref.LoadOp.get(op.memref.op.base_addr, op.indices)
                )
            else:
                raise ValueError("Failed to resolve memref.load called on dsd type")


class LowerSubviewOpPass(RewritePattern):
    """Lowers memref.subview to dsd ops"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
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
        subview: memref.SubviewOp, curr_op: SSAValue | Operation
    ) -> list[Operation]:
        assert isa(subview.source.type, MemRefType[Attribute])
        ops = list[Operation]()

        if subview.static_sizes.data.data[0].data == memref.SubviewOp.DYNAMIC_INDEX:
            ops.append(cast_op := arith.IndexCastOp(subview.sizes[0], csl.u16_value))
            ops.append(
                curr_op := csl.SetDsdLengthOp.build(
                    operands=[curr_op, cast_op], result_types=[subview.source.type]
                )
            )
        elif subview.static_sizes.as_tuple() != subview.source.type.get_shape():
            # update sizes only if they differ from op.source.type
            ops.append(
                len_op := arith.ConstantOp(
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
        subview: memref.SubviewOp, curr_op: SSAValue | Operation
    ) -> list[Operation]:
        assert isa(subview.source.type, MemRefType[Attribute])
        ops = list[Operation]()

        if subview.static_strides.data.data[0].data == memref.SubviewOp.DYNAMIC_INDEX:
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
                stride_op := arith.ConstantOp(
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
        subview: memref.SubviewOp, curr_op: SSAValue | Operation
    ) -> list[Operation]:
        assert isa(subview.source.type, MemRefType[Attribute])
        ops = list[Operation]()

        if subview.static_offsets.data.data[0].data == memref.SubviewOp.DYNAMIC_INDEX:
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
                offset_op := arith.ConstantOp(
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


class LowerUnrealizedConversionCastOpPass(RewritePattern):
    """
    Conversions from dsd to memref are no longer necessary after this pass.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: UnrealizedConversionCastOp, rewriter: PatternRewriter, /
    ):
        if all(isa(t, csl.DsdType) for t in op.inputs.types) and all(
            isa(t, MemRefType[Attribute]) for t in op.outputs.types
        ):
            rewriter.replace_matched_op([], new_results=op.inputs)


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


class CslVarUpdate(RewritePattern):
    """Update CSL Variable Definitions."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.VariableOp, rewriter: PatternRewriter, /):
        if (
            not isinstance(op.res.type, csl.VarType)
            or not isa(elem_t := op.res.type.get_element_type(), MemRefType[Attribute])
            or op.default
        ):
            return
        dsd_t = csl.DsdType(
            csl.DsdKind.mem1d_dsd if len(elem_t.shape) == 1 else csl.DsdKind.mem4d_dsd
        )
        rewriter.replace_matched_op(csl.VariableOp.from_type(dsd_t))


class CslVarLoad(RewritePattern):
    """Update CSL Load Variables."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.LoadVarOp, rewriter: PatternRewriter, /):
        if (
            not isa(op.res.type, MemRefType[Attribute])
            or not isinstance(op.var.type, csl.VarType)
            or not isa(op.var.type.get_element_type(), csl.DsdType)
        ):
            return
        rewriter.replace_matched_op(csl.LoadVarOp(op.var))


@dataclass(frozen=True)
class MemrefToDsdPass(ModulePass):
    """
    Lowers memref ops to CSL DSDs.

    Note, that CSL uses memref types in some places.

    This performs a backwards pass translating memref-consuming ops to dsd-consuming ops when all memref type
    information is known. A second forward pass translates memref-generating ops to dsd-generating ops.
    """

    name = "memref-to-dsd"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerSubviewOpPass(),
                    LowerCopyOpPass(),
                    LowerUnrealizedConversionCastOpPass(),
                ]
            ),
            walk_reverse=True,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
        forward_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CslVarUpdate(),
                    CslVarLoad(),
                    LowerAllocOpPass(),
                    DsdOpUpdateType(),
                    RetainAddressOfOpPass(),
                    FixMemrefLoadOnGetDsd(),
                    FixGetDsdOnGetDsd(),
                ]
            ),
            apply_recursively=False,
        )
        forward_pass.rewrite_module(op)
