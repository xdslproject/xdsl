from typing import cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    Float64Type,
    FloatAttr,
    ModuleOp,
)
from xdsl.ir import MLContext, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
)
from xdsl.sasha_rewrite_pattern import query_rewrite_pattern, rewrite_pattern_query
from xdsl.transforms.dead_code_elimination import dce
from xdsl.utils.hints import isa

from ..dialects.toy import ConstantOp, ReshapeOp, TensorTypeF64, TransposeOp


@rewrite_pattern_query
def simplify_redundant_transpose_query(root: TransposeOp, input: TransposeOp):
    return isa(root.arg, OpResult) and root.arg.op == input


@query_rewrite_pattern(simplify_redundant_transpose_query)
def simplify_redundant_transpose(
    rewriter: PatternRewriter, root: TransposeOp, input: TransposeOp
):
    rewriter.replace_matched_op((), (input.arg,))


@rewrite_pattern_query
def reshape_reshape_query(root: ReshapeOp, input: ReshapeOp):
    return isa(root.arg, OpResult) and root.arg.op == input


@query_rewrite_pattern(reshape_reshape_query)
def reshape_reshape(rewriter: PatternRewriter, root: ReshapeOp, input: ReshapeOp):
    t = cast(TensorTypeF64, root.res.type)
    new_op = ReshapeOp.from_input_and_type(input.arg, t)
    rewriter.replace_matched_op(new_op)


@rewrite_pattern_query
def fold_constant_reshape_query(root: ReshapeOp, input: ConstantOp):
    return isa(root.arg, OpResult) and root.arg.op == input


@query_rewrite_pattern(fold_constant_reshape_query)
def fold_constant_reshape(
    rewriter: PatternRewriter,
    root: ReshapeOp,
    input: ConstantOp,
):
    assert isa(root.res.type, TensorTypeF64)
    assert isa(input.value.data, ArrayAttr[FloatAttr[Float64Type]])
    new_value = DenseIntOrFPElementsAttr([root.res.type, input.value.data])
    new_op = ConstantOp(new_value)
    rewriter.replace_matched_op(new_op)


class OptimiseToy(ModulePass):
    name = "optimise-toy"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(simplify_redundant_transpose).rewrite_module(op)
        PatternRewriteWalker(reshape_reshape).rewrite_module(op)
        PatternRewriteWalker(fold_constant_reshape).rewrite_module(op)
        dce(op)
