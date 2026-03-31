import collections
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, csl, memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AffineMapAttr,
    ArrayAttr,
    Float16Type,
    Float32Type,
    IntAttr,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    NoneAttr,
    StridedLayoutAttr,
    UnrealizedConversionCastOp,
    i8,
    i16,
)
from xdsl.dialects.csl.csl import ZerosOpAttr
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr, AffineMap
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
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter, /):
        assert (
            MemRefType[ZerosOpAttr]
            .constr(csl.ZerosOpAttrConstr)
            .verifies(memref_type := op.memref.type)
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

        rewriter.replace_op(op, [zeros_op, *shape, dsd_op])


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
                rewriter.replace_op(
                    op,
                    csl.GetMemDsdOp.build(
                        operands=[op.base_addr.op.base_addr, op.sizes],
                        properties=op.properties,
                        attributes=op.attributes,
                        result_types=op.result_types,
                    ),
                )
            else:
                raise ValueError("Failed to resolve GetMemDsdOp called on dsd type")


class FixMemRefLoadOnGetDsd(RewritePattern):
    """
    MemRef load ops should load from the underlying memref, not from the dsd.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        if isinstance(op.memref.type, csl.DsdType):
            if isinstance(op.memref, OpResult) and isinstance(
                op.memref.op, csl.GetMemDsdOp
            ):
                rewriter.replace_op(
                    op, memref.LoadOp.get(op.memref.op.base_addr, op.indices)
                )
            else:
                raise ValueError("Failed to resolve memref.load called on dsd type")


class LowerSubviewOpPass(RewritePattern):
    """Lowers memref.subview to dsd ops"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        assert isa(op.source.type, MemRefType)
        assert isa(op.result.type, MemRefType)

        if len(op.result.type.get_shape()) == 1 and len(op.source.type.get_shape()) > 1:
            # 1d subview onto a nd memref
            sizes = op.static_sizes.get_values()
            counter_sizes = collections.Counter(sizes)
            counter_sizes.pop(1, None)
            assert len(counter_sizes) == 1, (
                "1d access into nd memref must specify one size > 1"
            )
            size, size_count = counter_sizes.most_common()[0]

            assert size_count == 1, (
                "1d access into nd memref can only specify one size > 1, which can occur only once"
            )
            assert all(stride == 1 for stride in op.static_strides.get_values()), (
                "All strides must equal 1"
            )

            amap: list[AffineExpr] = [
                AffineConstantExpr(o if o != DYNAMIC_INDEX else 0)
                for o in op.static_offsets.get_values()
            ]
            amap[sizes.index(size)] += AffineDimExpr(0)

            size_op = arith.ConstantOp.from_int_and_width(size, 16)
            dsd_op = csl.GetMemDsdOp(
                operands=[op.source, [size_op]],
                properties={
                    "tensor_access": AffineMapAttr(AffineMap(1, 0, tuple(amap)))
                },
                result_types=[csl.DsdType(csl.DsdKind.mem1d_dsd)],
            )
            offset_ops = self._update_offsets(op, dsd_op) if op.offsets else []
            rewriter.replace_op(op, [size_op, dsd_op, *offset_ops])
            return

        assert len(op.static_sizes) == 1, "not implemented"
        assert len(op.static_offsets) == 1, "not implemented"
        assert len(op.static_strides) == 1, "not implemented"

        last_op = op.source
        size_ops = self._update_sizes(op, last_op)

        last_op = size_ops[-1] if len(size_ops) > 0 else last_op
        stride_ops = self._update_strides(op, last_op)

        last_op = stride_ops[-1] if len(stride_ops) > 0 else last_op
        offset_ops = self._update_offsets(op, last_op)

        new_ops = [*size_ops, *stride_ops, *offset_ops]
        if new_ops:
            rewriter.replace_op(op, [*size_ops, *stride_ops, *offset_ops])
        else:
            # subview has no effect (todo: this could be canonicalized away)
            rewriter.replace_op(op, [], new_results=[op.source])

    @staticmethod
    def _update_sizes(
        subview: memref.SubviewOp, curr_op: SSAValue | Operation
    ) -> list[Operation]:
        assert isa(subview.source.type, MemRefType)
        ops = list[Operation]()

        static_sizes = subview.static_sizes.get_values()

        if static_sizes[0] == DYNAMIC_INDEX:
            ops.append(cast_op := arith.IndexCastOp(subview.sizes[0], i16))
            ops.append(
                curr_op := csl.SetDsdLengthOp.build(
                    operands=[curr_op, cast_op], result_types=[subview.source.type]
                )
            )
        elif static_sizes != subview.source.type.get_shape():
            # update sizes only if they differ from op.source.type
            ops.append(
                len_op := arith.ConstantOp(
                    IntegerAttr(
                        static_sizes[0],
                        i16,
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
        assert isa(subview.source.type, MemRefType)
        ops = list[Operation]()

        static_strides = subview.static_strides.get_values()

        if static_strides[0] == DYNAMIC_INDEX:
            ops.append(cast_op := arith.IndexCastOp(subview.strides[0], i8))
            ops.append(
                csl.SetDsdStrideOp.build(
                    operands=[curr_op, cast_op], result_types=[subview.source.type]
                )
            )
        elif static_strides != subview.source.type.get_strides():
            # update strides only if they differ from op.source.type
            ops.append(
                stride_op := arith.ConstantOp(
                    IntegerAttr(
                        static_strides[0],
                        i8,
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
        assert isa(subview.source.type, MemRefType)
        ops = list[Operation]()

        static_offsets = subview.static_offsets.get_values()

        if subview.offsets:
            ops.append(cast_op := arith.IndexCastOp(subview.offsets[0], i16))
            ops.append(
                csl.IncrementDsdOffsetOp.build(
                    operands=[curr_op, cast_op],
                    properties={"elem_type": subview.source.type.get_element_type()},
                    result_types=[subview.source.type],
                )
            )
        elif (
            isinstance(subview.source.type.layout, StridedLayoutAttr)
            and static_offsets[0] != (subview.source.type.layout.get_offset() or 0)
            or isinstance(subview.source.type.layout, NoneAttr)
            and static_offsets[0] != 0
        ):
            # update offsets only if they differ from op.source.type
            ops.append(
                offset_op := arith.ConstantOp(
                    IntegerAttr(
                        static_offsets[0],
                        i16,
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
        assert isa(op.source.type, MemRefType)

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

        rewriter.replace_op(op, func(operands=[[op.destination, op.source]]))


class LowerUnrealizedConversionCastOpPass(RewritePattern):
    """
    Conversions from dsd to memref are no longer necessary after this pass.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: UnrealizedConversionCastOp, rewriter: PatternRewriter, /
    ):
        if all(isa(t, csl.DsdType) for t in op.inputs.types) and all(
            isa(t, MemRefType) for t in op.outputs.types
        ):
            rewriter.replace_op(op, [], new_results=op.inputs)


class DsdOpUpdateType(RewritePattern):
    """Rebuild DSD ops from memref to DSD types."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: csl.IncrementDsdOffsetOp | csl.SetDsdStrideOp | csl.SetDsdLengthOp,
        rewriter: PatternRewriter,
        /,
    ):
        rewriter.replace_op(
            op,
            type(op).build(
                operands=op.operands,
                properties=op.properties,
                attributes=op.attributes,
                result_types=[op.op.type],
            ),
        )


class RetainAddressOfOpPass(RewritePattern):
    """Ensure we don't export DSD but the underlying memref."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.AddressOfOp, rewriter: PatternRewriter, /):
        if isinstance(op.value.type, csl.DsdType) and isinstance(
            op.value.owner, csl.GetMemDsdOp
        ):
            rewriter.replace_op(
                op,
                csl.AddressOfOp.build(
                    operands=[op.value.owner.base_addr], result_types=op.result_types
                ),
            )


class CslVarUpdate(RewritePattern):
    """Update CSL Variable Definitions."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.VariableOp, rewriter: PatternRewriter, /):
        if not isa(elem_t := op.res.type.get_element_type(), MemRefType) or op.default:
            return
        dsd_t = csl.DsdType(
            csl.DsdKind.mem1d_dsd if len(elem_t.shape) == 1 else csl.DsdKind.mem4d_dsd
        )
        rewriter.replace_op(op, csl.VariableOp.from_type(dsd_t))


class CslVarLoad(RewritePattern):
    """Update CSL Load Variables."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.LoadVarOp, rewriter: PatternRewriter, /):
        if (
            not isa(op.res.type, MemRefType)
            or not isinstance(op.var.type, csl.VarType)
            or not isa(op.var.type.get_element_type(), csl.DsdType)
        ):
            return
        rewriter.replace_op(op, csl.LoadVarOp(op.var))


@dataclass(frozen=True)
class MemRefToDsdPass(ModulePass):
    """
    Lowers memref ops to CSL DSDs.

    Note, that CSL uses memref types in some places.

    This performs a backwards pass translating memref-consuming ops to dsd-consuming ops when all memref type
    information is known. A second forward pass translates memref-generating ops to dsd-generating ops.
    """

    name = "memref-to-dsd"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
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
                    FixMemRefLoadOnGetDsd(),
                    FixGetDsdOnGetDsd(),
                ]
            ),
            apply_recursively=False,
        )
        forward_pass.rewrite_module(op)
