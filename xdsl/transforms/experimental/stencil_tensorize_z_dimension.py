from dataclasses import dataclass
from typing import Any, TypeGuard

from xdsl.context import Context
from xdsl.dialects import builtin, varith
from xdsl.dialects.arith import (
    ConstantOp,
    FloatingPointLikeBinaryOperation,
)
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    ContainerType,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IndexType,
    IntegerType,
    ModuleOp,
    ShapedType,
    TensorType,
)
from xdsl.dialects.csl import csl_stencil
from xdsl.dialects.experimental import dmp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.linalg import FillOp
from xdsl.dialects.stencil import (
    AccessOp,
    AccessPattern,
    ApplyOp,
    FieldType,
    IndexAttr,
    LoadOp,
    ReturnOp,
    StencilBoundsAttr,
    StoreOp,
    TempType,
)
from xdsl.dialects.tensor import EmptyOp, ExtractSliceOp, InsertSliceOp
from xdsl.ir import (
    Attribute,
    Block,
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.irdl import Operand
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


def get_required_result_type(op: Operation) -> TensorType[Any] | None:
    for result in op.results:
        for use in result.uses:
            if (
                isinstance(use.operation, ReturnOp)
                and (p_op := use.operation.parent_op()) is not None
            ):
                for ret in p_op.results:
                    if is_tensorized(ret.type):
                        if isa(ret.type, TempType) and isa(
                            r_type := ret.type.get_element_type(), TensorType
                        ):
                            return r_type
                # abort when encountering an un-tensorized ReturnOp successor
                return None
            if isinstance(use.operation, InsertSliceOp) and is_tensor(
                use.operation.result.type
            ):
                static_sizes = use.operation.static_sizes.get_values()
                assert is_tensor(use.operation.source.type)
                # inserting an (n-1)d tensor into an (n)d tensor should not require the input tensor to also be (n)d
                # instead, drop the first `dimdiff` dimensions
                dimdiff = len(static_sizes) - len(use.operation.source.type.shape)
                return TensorType(
                    use.operation.result.type.get_element_type(),
                    static_sizes[dimdiff:],
                )
            for ret in use.operation.results:
                if isa(r_type := ret.type, TensorType):
                    return r_type


def needs_update_shape(
    op_type: Attribute, succ_req_type: TensorType[Attribute]
) -> bool:
    assert isa(op_type, TensorType)
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


class StencilTypeConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: FieldType[Attribute]) -> FieldType[Attribute]:
        return stencil_field_to_tensor(typ)


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
        a = AccessOp.get(op.temp, xy_offsets)
        # this conditional controls if ExtractSliceOps for x/y accesses should be generated
        # if xy_offsets[0] != 0 or xy_offsets[1] != 0:
        #     rewriter.replace_op(op, a)
        #     return
        assert isa(op.temp.type, TempType[Attribute])
        assert is_tensor(element_t := op.temp.type.get_element_type())
        extract = ExtractSliceOp.from_static_parameters(
            a, [z_offset], element_t.get_shape()
        )
        rewriter.insert_op(a, InsertPoint.before(op))
        rewriter.replace_op(op, extract)


class ArithOpTensorize(RewritePattern):
    """
    Tensorises arith binary ops.
    If both operands are tensor types, rebuilds the op with matching result type.
    If one operand is scalar and an `arith.constant`, create a tensor constant directly.
    If one operand is scalar and not an `arith.constant`, create an empty tensor and fill it with the scalar value.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: FloatingPointLikeBinaryOperation, rewriter: PatternRewriter, /
    ):
        type_constructor = type(op)
        if is_tensor(op.result.type):
            return
        if is_tensor(op.lhs.type) and is_tensor(op.rhs.type):
            rewriter.replace_op(
                op,
                type_constructor(op.lhs, op.rhs, flags=None, result_type=op.lhs.type),
            )
        elif isa(op.lhs.type, TensorType[AnyFloat]) and is_scalar(op.rhs.type):
            new_rhs = ArithOpTensorize._rewrite_scalar_operand(
                op.rhs, op.lhs.type, op, rewriter
            )
            rewriter.replace_op(
                op,
                type_constructor(op.lhs, new_rhs, flags=None, result_type=op.lhs.type),
            )
        elif is_scalar(op.lhs.type) and isa(op.rhs.type, TensorType[AnyFloat]):
            new_lhs = ArithOpTensorize._rewrite_scalar_operand(
                op.lhs, op.rhs.type, op, rewriter
            )
            rewriter.replace_op(
                op,
                type_constructor(new_lhs, op.rhs, flags=None, result_type=op.rhs.type),
            )

    @staticmethod
    def _rewrite_scalar_operand(
        scalar_op: SSAValue,
        dest_typ: TensorType[AnyFloat],
        op: FloatingPointLikeBinaryOperation,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        """
        Rewrites a scalar operand into a tensor.
        If it is a constant, create a corresponding tensor constant.
        If it is not a constant, create an empty tensor and `linalg.fill` it with the scalar value.
        """
        if isinstance(scalar_op, OpResult) and isinstance(scalar_op.op, ConstantOp):
            assert isinstance(float_attr := scalar_op.op.value, FloatAttr)
            scalar_value = float_attr.value.data
            tens_const = ConstantOp(
                DenseIntOrFPElementsAttr.from_list(dest_typ, [scalar_value])
            )
            rewriter.insert_op(tens_const, InsertPoint.before(scalar_op.op))
            return tens_const.result
        emptyop = EmptyOp((), dest_typ)
        fillop = FillOp((scalar_op,), (emptyop.tensor,), (dest_typ,))
        rewriter.insert_op(emptyop, InsertPoint.before(op))
        rewriter.insert_op(fillop, InsertPoint.before(op))
        return fillop.res[0]


@dataclass(frozen=True)
class ApplyOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        if all(is_tensorized(arg.type) for arg in op.args):
            access_patterns = dict[Operand, AccessPattern](
                zip(op.region.block.args, op.get_accesses())
            )
            for access_op in op.region.walk():
                if isinstance(access_op, AccessOp):
                    z_shift = -access_patterns[access_op.temp].halo_in_axis(2)[0]
                    access_op.offset = IndexAttr.get(
                        *access_op.offset.array.data[:-1],
                        access_op.offset.array.data[-1].data + z_shift,
                    )

            body = Block(arg_types=op.operand_types)
            rewriter.inline_block(
                op.region.block, InsertPoint.at_start(body), body.args
            )

            rewriter.replace_op(
                op,
                ApplyOp.get(
                    op.args,
                    body,
                    [stencil_temp_to_tensor(r.type) for r in op.res],
                ),
            )


class FuncOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        if not op.is_declaration:
            for arg in op.args:
                if isa(arg.type, FieldType[Attribute]):
                    op.replace_argument_type(
                        arg, stencil_field_to_tensor(arg.type), rewriter
                    )


def is_tensorized(
    typ: Attribute,
):
    assert isinstance(typ, ShapedType)
    assert isinstance(typ, ContainerType)
    return len(typ.get_shape()) == 2 and isinstance(typ.get_element_type(), TensorType)


def is_tensor(
    typ: Attribute,
) -> TypeGuard[TensorType[IndexType | IntegerType | AnyFloat]]:
    return isinstance(typ, TensorType)


def is_scalar(typ: Attribute) -> TypeGuard[AnyFloat]:
    return isinstance(typ, AnyFloat)


class LoadOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        assert isa(op.res.type, TempType[Attribute])
        assert isinstance(bounds := op.res.type.bounds, StencilBoundsAttr)
        rewriter.replace_op(
            op,
            LoadOp.get(
                op.field,
                IndexAttr.get(*[lb for lb in bounds.lb][:-1]),
                IndexAttr.get(*[ub for ub in bounds.ub][:-1]),
            ),
        )


class DmpSwapOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        if (
            is_tensorized(op.input_stencil.type)
            and op.swapped_values
            and not is_tensorized(op.swapped_values.type)
        ):
            rewriter.replace_op(
                op,
                dmp.SwapOp.get(op.input_stencil, op.strategy, ArrayAttr(op.swaps.data)),
            )


class StoreOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        if (
            is_tensorized(op.field.type)
            and isinstance(op.field.type, ShapedType)
            and len(op.bounds.lb) != len(op.field.type.get_shape())
        ):
            rewriter.replace_op(
                op,
                StoreOp.get(
                    op.temp,
                    op.field,
                    StencilBoundsAttr(
                        zip(list(op.bounds.lb.array)[:-1], list(op.bounds.ub)[:-1])
                    ),
                ),
            )


class AccessOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.res.type, typ):
                rewriter.replace_op(
                    op,
                    AccessOp.build(
                        operands=[op.temp], attributes=op.attributes, result_types=[typ]
                    ),
                )


class CslStencilAccessOpUpdateShape(RewritePattern):
    """
    Updates the result type of a tensorized `csl_stencil.access` op
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.AccessOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.result.type, typ) and (
                isa(op.op.type, TempType[TensorType[Attribute]])
                or isa(op.op.type, TensorType[Attribute])
            ):
                rewriter.replace_op(
                    op,
                    csl_stencil.AccessOp(
                        op.op,
                        op.offset,
                        (
                            op.op.type.get_element_type()
                            if isa(op.op.type, TempType[TensorType[Attribute]])
                            else typ
                        ),
                        op.offset_mapping,
                    ),
                )


class ExtractSliceOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExtractSliceOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.result.type, typ):
                rewriter.replace_op(
                    op,
                    ExtractSliceOp.from_static_parameters(
                        op.source, op.static_offsets.get_values(), typ.get_shape()
                    ),
                )


def arithBinaryOpUpdateShape(
    op: FloatingPointLikeBinaryOperation,
    rewriter: PatternRewriter,
    /,
):
    type_constructor = type(op)
    if typ := get_required_result_type(op):
        if needs_update_shape(op.result.type, typ):
            rewriter.replace_op(op, type_constructor(op.lhs, op.rhs, result_type=typ))


class ArithOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: FloatingPointLikeBinaryOperation, rewriter: PatternRewriter, /
    ):
        arithBinaryOpUpdateShape(op, rewriter)


class VarithOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: varith.VarithOp, rewriter: PatternRewriter, /):
        type_constructor = type(op)
        if typ := get_required_result_type(op):
            if needs_update_shape(op.result_types[0], typ):
                rewriter.replace_op(
                    op, type_constructor.build(operands=[op.args], result_types=[typ])
                )


class EmptyOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: EmptyOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.results[0].type, typ):
                rewriter.replace_op(op, EmptyOp((), typ))


class FillOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FillOp, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.results[0].type, typ):
                rewriter.replace_op(
                    op, FillOp(op.inputs, op.outputs, [typ] * len(op.outputs))
                )


class ConstOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter, /):
        if is_tensor(op.result.type):
            if typ := get_required_result_type(op):
                if needs_update_shape(op.result.type, typ):
                    assert isinstance(op.value, DenseIntOrFPElementsAttr)
                    rewriter.replace_op(
                        op, ConstantOp(DenseIntOrFPElementsAttr(typ, op.value.data))
                    )


@dataclass(frozen=True)
class BackpropagateStencilShapes(ModulePass):
    """
    Greedily back-propagates the result types of tensorized ops.
    Use after creating/modifying tensorization.
    """

    name = "backpropagate-stencil-shapes"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        backpropagate_stencil_shapes = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CslStencilAccessOpUpdateShape(),
                    ExtractSliceOpUpdateShape(),
                    EmptyOpUpdateShape(),
                    FillOpUpdateShape(),
                    ArithOpUpdateShape(),
                    VarithOpUpdateShape(),
                    ConstOpUpdateShape(),
                ]
            ),
            walk_reverse=True,
            apply_recursively=False,
        )
        backpropagate_stencil_shapes.rewrite_module(op)


@dataclass(frozen=True)
class StencilTensorizeZDimension(ModulePass):
    name = "stencil-tensorize-z-dimension"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FuncOpTensorize(),
                    StencilTypeConversion(),  # this needs to come after FuncOpTensorize()
                    LoadOpTensorize(),
                    ApplyOpTensorize(),
                    StoreOpTensorize(),
                    DmpSwapOpTensorize(),
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
                    ArithOpTensorize(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        stencil_pass.rewrite_module(op)
        BackpropagateStencilShapes().apply(ctx=ctx, op=op)
