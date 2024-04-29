from typing import cast

from attr import dataclass

from xdsl.dialects.bufferization import ToTensorOp
from xdsl.dialects.builtin import (
    ContainerType,
    MemRefType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.func import FuncOp
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    ExternalLoadOp,
    FieldType,
    LoadOp,
    StencilBoundsAttr,
    StencilType,
    TempType,
)
from xdsl.dialects.tensor import ExtractSliceOp
from xdsl.ir import (
    Attribute,
    MLContext,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)


def stencil_field_to_tensor(field: StencilType[Attribute]):
    if field.get_num_dims() != 3:
        return field
    typ = TensorType(field.get_element_type(), [field.get_shape()[-1]])
    bounds = [
        (field.bounds.lb.array.data[i], field.bounds.ub.array.data[i])
        for i in range(field.get_num_dims() - 1)
    ]
    return FieldType[Attribute](bounds, typ)


def stencil_temp_to_tensor(field: StencilType[Attribute]):
    if field.get_num_dims() != 3:
        return field
    typ = TensorType(field.get_element_type(), [field.get_shape()[-1]])
    bounds = [
        (field.bounds.lb.array.data[i], field.bounds.ub.array.data[i])
        for i in range(field.get_num_dims() - 1)
    ]
    return TempType[Attribute](bounds, typ)


def stencil_memref_to_tensor(field: MemRefType[Attribute]):
    if field.get_num_dims() != 3:
        return field
    typ = TensorType(field.get_element_type(), [field.get_shape()[-1]])
    return MemRefType[Attribute](typ, field.get_shape()[:-1])


class StencilFieldConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: FieldType[Attribute]) -> FieldType[Attribute]:
        return stencil_field_to_tensor(typ)


class StencilTempConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: TempType[Attribute]) -> TempType[Attribute]:
        return stencil_temp_to_tensor(typ)


class StencilMemRefConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: MemRefType[Attribute]) -> MemRefType[Attribute]:
        return stencil_memref_to_tensor(typ)


class LoadOpToExtractSlice(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        to_tensor = ToTensorOp(op.field, writable=True, restrict=True)
        field_t = cast(StencilType[Attribute], op.field.type)
        temp_t = cast(StencilType[Attribute], op.res.type)
        assert isinstance(field_t.bounds, StencilBoundsAttr)
        assert isinstance(temp_t.bounds, StencilBoundsAttr)
        offsets = tuple(
            -flb + tlb for flb, tlb in zip(field_t.bounds.lb, temp_t.bounds.lb)
        )
        sizes = temp_t.get_shape()
        extract = ExtractSliceOp.from_static_parameters(
            to_tensor.tensor, offsets, sizes
        )
        rewriter.replace_matched_op((to_tensor, extract))


@dataclass(frozen=True)
class AccessOpToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        # assert len(op.offset) == 3
        if len(op.offset) != 3:
            return
        xy_offsets, _ = (
            tuple(o for o in op.offset)[:-1],
            tuple(o for o in op.offset)[-1],
        )
        a = AccessOp.get(op.temp, xy_offsets, op.offset_mapping)
        if xy_offsets[0] != 0 or xy_offsets[1] != 0:
            rewriter.replace_matched_op(a)
            return
        # TODO generate ExtractSliceOp
        # ExtractSliceOp.from_static_parameters(a, offsets, shape)
        # slice = ExtractSliceOp.from_static_parameters(a, (z_offset,), [(-1,511)])
        # rewriter.insert_op_after(slice)
        rewriter.replace_matched_op(a)


@dataclass(frozen=True)
class ApplyOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        # if (all(is_tensorised(arg.type) for arg in op.args) and not is_tensorised(op.result.type)):
        # if isinstance(op.res[0].type.get_element_type(), ContainerType):
        #     return
        # args = [stencil_temp_to_tensor(a.type) for a in op.args]
        # res = [stencil_temp_to_tensor(r.type) for r in op.res]
        # rewriter.replace_matched_op(ApplyOp.get(args, op.region, res))
        pass


class FuncOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        for arg in op.args:
            op.replace_argument_type(arg, stencil_memref_to_tensor(arg.type))


def is_tensorised(typ):
    return len(typ.get_shape()) == 2 and isinstance(typ, ContainerType)


class ExternalLoadOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        if is_tensorised(op.field.type) and not is_tensorised(op.result.type):
            rewriter.replace_matched_op(
                ExternalLoadOp.get(op.field, stencil_field_to_tensor(op.result.type))
            )


class LoadOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        if is_tensorised(op.field.type) and not is_tensorised(op.res.type):
            rewriter.replace_matched_op(
                LoadOp.get(
                    op.field,
                    [lb for lb in op.res.type.bounds.lb][:-1],
                    [ub for ub in op.res.type.bounds.ub][:-1],
                )
            )


@dataclass(frozen=True)
class StencilTensorizeDimension(ModulePass):
    name = "stencil-tensorize-dimension"
    # dimension: int

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # ctx.get_optional_op("bufferization.materialize_in_destination")
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FuncOpTensorize(),
                    ExternalLoadOpTensorize(),
                    LoadOpTensorize(),
                    ApplyOpTensorize(),
                    # BufferOpToAlloc(),
                    # ExternalLoadToUnerealizedConversionCast(),
                    # TrivialExternalStoreOpCleanup(),
                    # StoreOpToInsertSlice(),
                    # LoadOpToExtractSlice(),
                    # CastOpToCast(),
                    # ReturnOpToYield(),
                    # AccessOpToTensor(),
                ]
            ),
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    StencilTempConversion(recursive=True),
                    StencilFieldConversion(recursive=True),
                    StencilMemRefConversion(recursive=True),
                ]
            )
        )
        # type_conversion_pass.rewrite_module(op)
