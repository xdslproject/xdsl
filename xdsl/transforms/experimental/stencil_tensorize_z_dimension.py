from attr import dataclass

from xdsl.dialects.arith import Addf, BinaryOperation, Divf, Mulf, Subf
from xdsl.dialects.builtin import (
    ContainerType,
    MemRefType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.func import FuncOp
from xdsl.dialects.linalg import FillOp
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    ExternalLoadOp,
    FieldType,
    LoadOp,
    ReturnOp,
    StencilType,
    TempType,
)
from xdsl.dialects.tensor import EmptyOp, ExtractSliceOp
from xdsl.ir import (
    Attribute,
    MLContext,
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


def get_required_result_type(op: Operand) -> TensorType | None:
    for result in op.results:
        for use in result.uses:
            if isinstance(use.operation, ReturnOp):
                for ret in use.operation.parent_op().results:
                    if is_tensorized(ret.type):
                        return ret.type.get_element_type()
                # abort when encountering an un-tensorized ReturnOp successor
                return None
            for ret in use.operation.results:
                if is_tensor(ret.type):
                    return ret.type


def needs_update_shape(op_type: TensorType, succ_req_type: TensorType) -> bool:
    return op_type.get_shape() != succ_req_type.get_shape()


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
        extract = ExtractSliceOp.from_static_parameters(
            a, [z_offset], op.temp.type.get_element_type().get_shape()
        )
        rewriter.insert_op_before(a, op)
        rewriter.replace_matched_op(extract)


class ArithOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BinaryOperation, rewriter: PatternRewriter, /):
        if is_tensor(op.result.type):
            return
        ctor = None
        if isinstance(op, Addf):
            ctor = Addf
        elif isinstance(op, Subf):
            ctor = Subf
        elif isinstance(op, Mulf):
            ctor = Mulf
        elif isinstance(op, Divf):
            ctor = Divf
        if is_tensor(op.lhs.type) and is_tensor(op.rhs.type):
            rewriter.replace_matched_op(
                ctor(op.lhs, op.rhs, flags=None, result_type=op.lhs.type)
            )
        elif is_tensor(op.lhs.type) and is_scalar(op.rhs.type):
            emptyop = EmptyOp((), op.lhs.type)
            fillop = FillOp((op.rhs,), (emptyop,), (op.lhs.type,))
            rewriter.insert_op_before(emptyop, op)
            rewriter.insert_op_before(fillop, op)
            rewriter.replace_matched_op(
                ctor(op.lhs, fillop, flags=None, result_type=op.lhs.type)
            )
        elif is_scalar(op.lhs.type) and is_tensor(op.rhs.type):
            emptyop = EmptyOp((), op.rhs.type)
            fillop = FillOp((op.lhs,), (emptyop,), (op.rhs.type,))
            rewriter.insert_op_before(emptyop, op)
            rewriter.insert_op_before(fillop, op)
            rewriter.replace_matched_op(
                ctor(fillop, op.rhs, flags=None, result_type=op.rhs.type)
            )


class ReturnOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        if all(is_tensorized(r.type) for r in op.parent_op().res):
            pass


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
                arg = b.insert_arg(stencil_temp_to_tensor(b.args[0].type), len(b.args))
                b.args[0].replace_by(arg)
                b.erase_arg(b.args[0], safe_erase=True)
            rewriter.replace_matched_op(
                ApplyOp.get(
                    op.args,
                    op.region.clone(),
                    [stencil_temp_to_tensor(r.type) for r in op.res],
                )
            )


class FuncOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        for arg in op.args:
            op.replace_argument_type(arg, stencil_memref_to_tensor(arg.type))


def is_tensorized(typ):
    return len(typ.get_shape()) == 2 and isinstance(typ, ContainerType)


def is_tensor(typ):
    return isinstance(typ, TensorType)


def is_scalar(typ):
    return not isinstance(typ, ContainerType)


class ExternalLoadOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        if is_tensorized(op.field.type) and not is_tensorized(op.result.type):
            rewriter.replace_matched_op(
                ExternalLoadOp.get(op.field, stencil_field_to_tensor(op.result.type))
            )


class LoadOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        if is_tensorized(op.field.type) and not is_tensorized(op.res.type):
            rewriter.replace_matched_op(
                LoadOp.get(
                    op.field,
                    [lb for lb in op.res.type.bounds.lb][:-1],
                    [ub for ub in op.res.type.bounds.ub][:-1],
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
                rewriter.replace_matched_op(
                    ExtractSliceOp.from_static_parameters(
                        op.source, op.static_offsets.data.data, typ.get_shape()
                    )
                )


class ArithOpUpdateShape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BinaryOperation, rewriter: PatternRewriter, /):
        if typ := get_required_result_type(op):
            if needs_update_shape(op.result.type, typ):
                ctor = None
                if isinstance(op, Addf):
                    ctor = Addf
                elif isinstance(op, Subf):
                    ctor = Subf
                elif isinstance(op, Mulf):
                    ctor = Mulf
                elif isinstance(op, Divf):
                    ctor = Divf
                rewriter.replace_matched_op(
                    ctor(op.lhs, op.rhs, flags=None, result_type=typ)
                )


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
        # ctx.get_optional_op("bufferization.materialize_in_destination")
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FuncOpTensorize(),
                    ExternalLoadOpTensorize(),
                    LoadOpTensorize(),
                    ApplyOpTensorize(),
                    # AccessOpTensorize(),   # these don't work here
                    # ArithOpTensorize(),    # use second pass
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
        backpropagate_stencil_shapes = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # AccessOpUpdateShape(),
                    ExtractSliceOpUpdateShape(),
                    EmptyOpUpdateShape(),
                    FillOpUpdateShape(),
                    ArithOpUpdateShape(),
                ]
            ),
            walk_reverse=True,
            apply_recursively=False,
        )
        backpropagate_stencil_shapes.rewrite_module(op)
