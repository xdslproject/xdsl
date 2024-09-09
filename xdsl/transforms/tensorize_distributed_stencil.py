from dataclasses import dataclass
from typing import TypeGuard

from xdsl.context import MLContext
from xdsl.dialects import func, stencil
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute
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
from xdsl.transforms.experimental.stencil_tensorize_z_dimension import ArithOpTensorize
from xdsl.utils.hints import isa
from xdsl.utils.isa import isattr


def is_tensorized(
    typ: Attribute,
) -> TypeGuard[stencil.StencilType[TensorType[Attribute]]]:
    return isa(typ, stencil.StencilType[TensorType[Attribute]])


class FuncOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        op.update_function_type()


class DmpSwapOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        if op.swapped_values and is_tensorized(op.swapped_values.type):
            return
        if isattr(op.input_stencil.type, stencil.FieldType):
            pass
        pass


class StoreOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        pass


@dataclass(frozen=True)
class ApplyOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ApplyOp, rewriter: PatternRewriter, /):
        pass


@dataclass(frozen=True)
class ReturnOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ReturnOp, rewriter: PatternRewriter, /):
        pass


@dataclass(frozen=True)
class AccessOpTensorize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.AccessOp, rewriter: PatternRewriter, /):
        if isa(op.temp.type, stencil.StencilType[TensorType[Attribute]]) and not isa(
            op.res.type, TensorType[Attribute]
        ):
            rewriter.replace_matched_op(
                stencil.AccessOp.get(op.temp, tuple(op.offset), op.offset_mapping)
            )


class StencilTypeConversion(TypeConversionPattern):
    tensor_shape: tuple[int, ...]

    def __init__(self, tensor_shape: tuple[int, ...]):
        self.tensor_shape = tensor_shape
        super().__init__()

    @attr_type_rewrite_pattern
    def convert_type(
        self, typ: stencil.StencilType[Attribute]
    ) -> stencil.StencilType[Attribute] | None:
        if isa(typ, stencil.StencilType[TensorType[Attribute]]):
            return None
        return type(typ)(
            typ.bounds, TensorType(typ.get_element_type(), self.tensor_shape)
        )


@dataclass(frozen=True)
class TensorizeDistributedStencilPass(ModulePass):
    name = "tensorize-distributed-stencil"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        decomp: tuple[int, ...] | None = None
        for curr_op in op.walk():
            if (
                (
                    isinstance(curr_op, stencil.StoreOp)
                    or isinstance(curr_op, stencil.ApplyOp)
                )
                and curr_op.bounds
                and isattr(curr_op.bounds, stencil.StencilBoundsAttr)
            ):
                # this assert is also made in the `distribute-stencil` pass
                assert all(
                    integer_attr.data == 0
                    for integer_attr in curr_op.bounds.lb.array.data
                ), "lb must be 0"
                curr_decomp: tuple[int, ...] = tuple(curr_op.bounds.ub)
                if decomp:
                    assert decomp == curr_decomp, "Encountered incompatible slices"
                else:
                    decomp = curr_decomp
        assert decomp, "Cannot perform tensorization"
        # decomp = tuple(d for d in decomp if d > 1) or (1,)

        PatternRewriteWalker(StencilTypeConversion(tensor_shape=decomp)).rewrite_module(
            op
        )

        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # StencilTypeConversion(tensor_shape=decomp),
                    ArithOpTensorize(),
                    AccessOpTensorize(),
                    FuncOpTensorize(),
                    # ExternalLoadOpTensorize(),
                    # LoadOpTensorize(),
                    # ApplyOpTensorize(),
                    # StoreOpTensorize(),
                    # ReturnOpTensorize(),
                    # DmpSwapOpTensorize(),
                ]
            ),
        )
        module_pass.rewrite_module(op)
