from typing import cast

from xdsl.ir import MLContext, OpResult
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ArrayAttr,
    Float64Type,
    FloatAttr,
    ModuleOp,
    TensorType,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, implicit_rewriter
from xdsl.transforms.dead_code_elimination import dce
from xdsl.utils.hints import isa

from ..dialects.toy import ConstantOp, ReshapeOp, TensorTypeF64, TransposeOp


@implicit_rewriter
def simplify_redundant_transpose(op: TransposeOp):
    """Fold transpose(transpose(x)) -> x"""
    if isinstance(input := op.arg, OpResult) and isinstance(input.op, TransposeOp):
        return input.op.operands


@implicit_rewriter
def reshape_reshape(op: ReshapeOp):
    """Reshape(Reshape(x)) = Reshape(x)"""
    if isinstance(input := op.arg, OpResult) and isinstance(input.op, ReshapeOp):
        t = cast(TensorType[Float64Type], op.res.typ)
        new_op = ReshapeOp(input.op.arg, t)
        return new_op.results


@implicit_rewriter
def constant_reshape(op: ReshapeOp):
    """
    Reshaping a constant can be done at compile time
    """
    if isinstance(input := op.arg, OpResult) and isinstance(input.op, ConstantOp):
        assert isa(op.res.typ, TensorTypeF64)
        assert isa(input.op.value.data, ArrayAttr[FloatAttr[Float64Type]])

        new_value = DenseIntOrFPElementsAttr.create_dense_float(
            type=op.res.typ, data=input.op.value.data.data
        )
        new_op = ConstantOp(new_value)
        return new_op.results


class OptimiseToy(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(simplify_redundant_transpose).rewrite_module(op)
        PatternRewriteWalker(reshape_reshape).rewrite_module(op)
        PatternRewriteWalker(constant_reshape).rewrite_module(op)
        dce(op)
