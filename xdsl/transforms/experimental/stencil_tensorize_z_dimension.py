from collections.abc import Callable, Sequence
from typing import TypeGuard, cast

from attr import dataclass

from xdsl.dialects.arith import Addf, Divf, FloatingPointLikeBinaryOp, Mulf, Subf
from xdsl.dialects.builtin import (
    AnyFloat,
    ContainerType,
    IntAttr,
    MemRefType,
    ModuleOp,
    ShapedType,
    TensorType,
)
from xdsl.dialects.func import FuncOp
from xdsl.dialects.linalg import FillOp
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    ExternalLoadOp,
    FieldType,
    IndexAttr,
    LoadOp,
    ReturnOp,
    StencilBoundsAttr,
    StoreOp,
    TempType,
)
from xdsl.dialects.tensor import EmptyOp, ExtractSliceOp
from xdsl.ir import (
    Attribute,
    MLContext,
    Operation,
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
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


def get_required_result_type(op: Operation) -> TensorType[Attribute] | None:
    for result in op.results:
        for use in result.uses:
            if (
                isinstance(use.operation, ReturnOp)
                and (p_op := use.operation.parent_op()) is not None
            ):
                for ret in p_op.results:
                    if is_tensorized(ret.type):
                        if isa(ret.type, TempType[Attribute]) and isa(
                            r_type := ret.type.get_element_type(), TensorType[Attribute]
                        ):
                            return r_type
                # abort when encountering an un-tensorized ReturnOp successor
                return None
            for ret in use.operation.results:
                if isa(r_type := ret.type, TensorType[Attribute]):
                    return r_type


def needs_update_shape(
    op_type: Attribute, succ_req_type: TensorType[Attribute]
) -> bool:
    assert isa(op_type, TensorType[Attribute])
    return op_type.get_shape() != succ_req_type.get_shape()


def stencil_field_to_tensor(field: FieldType[Attribute]) -> FieldType[Attribute]:
    if field.get_num_dims() != 3:
        return field
    typ = TensorType(field.get_element_type(), [field.get_shape()[-1]])
    assert isinstance(field.bounds, StencilBoundsAttr)
    assert isinstance(field.bounds.lb, IndexAttr)
    assert isinstance(field.bounds.ub, IndexAttr)
    bounds = list(zip(field.bounds.lb, field.bounds.ub))[:-1]
    return FieldType[Attribute](bounds, typ)


def stencil_temp_to_tensor(field: TempType[Attribute]) -> TempType[Attribute]:
    if field.get_num_dims() != 3:
        return field
    typ = TensorType(field.get_element_type(), [field.get_shape()[-1]])
    assert isinstance(field.bounds, StencilBoundsAttr)
    assert isinstance(field.bounds.lb, IndexAttr)
    assert isinstance(field.bounds.ub, IndexAttr)
    bounds = list(zip(field.bounds.lb, field.bounds.ub))[:-1]
    return TempType[Attribute](bounds, typ)


def stencil_memref_to_tensor(field: MemRefType[Attribute]) -> MemRefType[Attribute]:
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


@dataclass(frozen=True)
class AccessOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        if not is_tensorized(op.operands[0].type) or len(op.offset) != 3:
            return
        xy_offsets, z_offset = (
            tuple(o for o in op.offset)[:-1],
            tuple(o for o in op.offset)[-1],
        )
        a = AccessOp.get(op.temp, xy_offsets, op.offset_mapping)
        # this conditional controls if ExtractSliceOps for x/y accesses should be generated
        # if xy_offsets[0] != 0 or xy_offsets[1] != 0:
        #     rewriter.replace_matched_op(a)
        #     return
        assert isa(op.temp.type, TempType[Attribute])
        assert is_tensor(element_t := op.temp.type.get_element_type())
        extract = ExtractSliceOp.from_static_parameters(
            a, [z_offset], element_t.get_shape()
        )
        rewriter.insert_op(a, InsertPoint.before(op))
        rewriter.replace_matched_op(extract)


def arithBinaryOpTensorize(
    type_constructor: Callable[..., FloatingPointLikeBinaryOp],
    op: FloatingPointLikeBinaryOp,
    rewriter: PatternRewriter,
    /,
):
    if is_tensor(op.result.type):
        return
    if is_tensor(op.lhs.type) and is_tensor(op.rhs.type):
        rewriter.replace_matched_op(
            type_constructor(op.lhs, op.rhs, flags=None, result_type=op.lhs.type)
        )
    elif is_tensor(op.lhs.type) and is_scalar(op.rhs.type):
        emptyop = EmptyOp((), op.lhs.type)
        fillop = FillOp((op.rhs,), (emptyop,), (op.lhs.type,))
        rewriter.insert_op(emptyop, InsertPoint.before(op))
        rewriter.insert_op(fillop, InsertPoint.before(op))
        rewriter.replace_matched_op(
            type_constructor(op.lhs, fillop, flags=None, result_type=op.lhs.type)
        )
    elif is_scalar(op.lhs.type) and is_tensor(op.rhs.type):
        emptyop = EmptyOp((), op.rhs.type)
        fillop = FillOp((op.lhs,), (emptyop,), (op.rhs.type,))
        rewriter.insert_op(emptyop, InsertPoint.before(op))
        rewriter.insert_op(fillop, InsertPoint.before(op))
        rewriter.replace_matched_op(
            type_constructor(fillop, op.rhs, flags=None, result_type=op.rhs.type)
        )


class ArithAddfOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Addf, rewriter: PatternRewriter, /):
        arithBinaryOpTensorize(Addf, op, rewriter)


class ArithSubfOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Subf, rewriter: PatternRewriter, /):
        arithBinaryOpTensorize(Subf, op, rewriter)


class ArithMulfOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Mulf, rewriter: PatternRewriter, /):
        arithBinaryOpTensorize(Mulf, op, rewriter)


class ArithDivfOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Divf, rewriter: PatternRewriter, /):
        arithBinaryOpTensorize(Divf, op, rewriter)


@dataclass(frozen=True)
class ApplyOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        if all(is_tensorized(arg.type) for arg in op.args) and all(
            not is_tensorized(r.type) for r in op.res
        ):
            b = op.region.block
            # TODO check if block args need updating
            for _ in range(len(b.args)):
                assert isa(arg_type := b.args[0].type, TempType[Attribute])
                arg = b.insert_arg(stencil_temp_to_tensor(arg_type), len(b.args))
                b.args[0].replace_by(arg)
                b.erase_arg(b.args[0], safe_erase=True)
            rewriter.replace_matched_op(
                ApplyOp.get(
                    op.args,
                    op.region.clone(),
                    [
                        stencil_temp_to_tensor(cast(TempType[Attribute], r.type))
                        for r in op.res
                    ],
                )
            )


class FuncOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        for arg in op.args:
            assert isa(arg.type, MemRefType[Attribute])
            op.replace_argument_type(arg, stencil_memref_to_tensor(arg.type))


def is_tensorized(
    typ: Attribute,
):
    assert isinstance(typ, ShapedType)
    assert isinstance(typ, ContainerType)
    return len(typ.get_shape()) == 2 and isinstance(typ.get_element_type(), TensorType)


def is_tensor(typ: Attribute) -> TypeGuard[TensorType[Attribute]]:
    return isinstance(typ, TensorType)


def is_scalar(typ: Attribute) -> TypeGuard[AnyFloat]:
    return isinstance(typ, AnyFloat)


class ExternalLoadOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        if is_tensorized(op.field.type) and not is_tensorized(op.result.type):
            assert isa(op.result.type, FieldType[Attribute])
            rewriter.replace_matched_op(
                ExternalLoadOp.get(op.field, stencil_field_to_tensor(op.result.type))
            )


class LoadOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        if is_tensorized(op.field.type) and not is_tensorized(op.res.type):
            assert isa(op.res.type, TempType[Attribute])
            assert isinstance(bounds := op.res.type.bounds, StencilBoundsAttr)
            rewriter.replace_matched_op(
                LoadOp.get(
                    op.field,
                    IndexAttr.get(*[lb for lb in bounds.lb][:-1]),
                    IndexAttr.get(*[ub for ub in bounds.ub][:-1]),
                )
            )


class StoreOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        if (
            is_tensorized(op.field.type)
            and isinstance(op.field.type, ShapedType)
            and len(op.bounds.lb) != len(op.field.type.get_shape())
        ):
            rewriter.replace_matched_op(
                StoreOp.get(
                    op.temp,
                    op.field,
                    StencilBoundsAttr(
                        zip(list(op.bounds.lb.array)[:-1], list(op.bounds.ub)[:-1])
                    ),
                )
            )


class AccessOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.res.type, typ):
                rewriter.replace_matched_op(
                    AccessOp.build(
                        operands=[op.temp], attributes=op.attributes, result_types=[typ]
                    )
                )


class ExtractSliceOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExtractSliceOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.result.type, typ):
                # offsets = op.static_offsets.data.data
                if isa(offsets := op.static_offsets.data.data, Sequence[IntAttr]):
                    new_offsets = [o.data for o in offsets]
                else:
                    assert isa(offsets, Sequence[int])
                    new_offsets = offsets
                rewriter.replace_matched_op(
                    ExtractSliceOp.from_static_parameters(
                        op.source, new_offsets, typ.get_shape()
                    )
                )


def arithBinaryOpUpdateShape(
    type_constructor: Callable[..., FloatingPointLikeBinaryOp],
    op: FloatingPointLikeBinaryOp,
    rewriter: PatternRewriter,
    /,
):
    if typ := get_required_result_type(op):
        if needs_update_shape(op.result.type, typ):
            rewriter.replace_matched_op(
                type_constructor(op.lhs, op.rhs, flags=None, result_type=typ)
            )


class ArithAddfOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Addf, rewriter: PatternRewriter, /):
        arithBinaryOpUpdateShape(Addf, op, rewriter)


class ArithSubfOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Subf, rewriter: PatternRewriter, /):
        arithBinaryOpUpdateShape(Subf, op, rewriter)


class ArithMulfOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Mulf, rewriter: PatternRewriter, /):
        arithBinaryOpUpdateShape(Mulf, op, rewriter)


class ArithDivfOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Divf, rewriter: PatternRewriter, /):
        arithBinaryOpUpdateShape(Divf, op, rewriter)


class EmptyOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: EmptyOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.results[0].type, typ):
                rewriter.replace_matched_op(EmptyOp((), typ))


class FillOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FillOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.results[0].type, typ):
                rewriter.replace_matched_op(
                    FillOp(op.inputs, op.outputs, [typ] * len(op.outputs))
                )


@dataclass(frozen=True)
class StencilTensorizeZDimension(ModulePass):
    name = "stencil-tensorize-z-dimension"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FuncOpTensorize(),
                    ExternalLoadOpTensorize(),
                    LoadOpTensorize(),
                    ApplyOpTensorize(),
                    StoreOpTensorize(),
                    # AccessOpTensorize(),   # this doesn't work here, using second pass
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
        stencil_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AccessOpTensorize(),
                    ArithAddfOpTensorize(),
                    ArithMulfOpTensorize(),
                    ArithSubfOpTensorize(),
                    ArithDivfOpTensorize(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        stencil_pass.rewrite_module(op)
        backpropagate_stencil_shapes = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # AccessOpUpdateShape(),
                    ExtractSliceOpUpdateShape(),
                    EmptyOpUpdateShape(),
                    FillOpUpdateShape(),
                    ArithAddfOpUpdateShape(),
                    ArithSubfOpUpdateShape(),
                    ArithMulfOpUpdateShape(),
                    ArithDivfOpUpdateShape(),
                ]
            ),
            walk_reverse=True,
            apply_recursively=False,
        )
        backpropagate_stencil_shapes.rewrite_module(op)
