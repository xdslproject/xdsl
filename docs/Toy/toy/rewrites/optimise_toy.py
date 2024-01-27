from typing import cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    Float64Type,
    FloatAttr,
    TensorType,
)
from xdsl.ir import OpResult
from xdsl.pattern_rewriter import implicit_rewriter
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
        t = cast(TensorType[Float64Type], op.res.type)
        new_op = ReshapeOp(input.op.arg, t)
        return new_op.results


@implicit_rewriter
def fold_constant_reshape(op: ReshapeOp):
    """
    Reshaping a constant can be done at compile time
    """
    if isinstance(input := op.arg, OpResult) and isinstance(input.op, ConstantOp):
        assert isa(op.res.type, TensorTypeF64)
        assert isa(input.op.value.data, ArrayAttr[FloatAttr[Float64Type]])

        new_value = DenseIntOrFPElementsAttr.create_dense_float(
            type=op.res.type, data=input.op.value.data.data
        )
        new_op = ConstantOp(new_value)
        return new_op.results
