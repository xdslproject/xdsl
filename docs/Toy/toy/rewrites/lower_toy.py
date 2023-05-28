from typing import cast
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    UnrankedTensorType,
    TensorType,
    i32,
    Float64Type,
)
from xdsl.dialects import llvm
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, implicit_rewriter
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy as td
from ..dialects import vector as tvd

f64 = Float64Type()


@implicit_rewriter
def lower_tensor_constant(op: td.ConstantOp):
    typ = op.value.type

    assert isinstance(typ, TensorType), "Toy constants always have rank information"
    typ = cast(td.AnyTensorTypeF64, typ)

    shape = DenseIntOrFPElementsAttr.vector_from_list(op.get_shape(), i32)
    data = DenseIntOrFPElementsAttr.vector_from_list(op.get_data(), f64)

    shape_vector = tvd.VectorConstantOp(shape, "tensor_shape")
    data_vector = tvd.VectorConstantOp(data, "tensor_data")
    tensor = tvd.TensorMakeOp(shape_vector, data_vector, typ)

    return tensor.results


@implicit_rewriter
def lower_print_op(op: td.PrintOp):
    return llvm.CallIntrinsicOp("tensor.print", (op.input,), ()).results


@implicit_rewriter
def lower_reshape(op: td.ReshapeOp):
    typ = op.res.typ
    assert isinstance(typ, TensorType)
    typ = cast(td.TensorTypeF64, typ)
    shape = DenseIntOrFPElementsAttr.vector_from_list(typ.get_shape(), i32)

    new_shape = tvd.VectorConstantOp(shape, "tensor_new_shape")
    old_data = tvd.TensorDataOp(op.arg)
    make_tensor = tvd.TensorMakeOp(new_shape, old_data, typ)

    return make_tensor.results


@implicit_rewriter
def lower_tensor_add(op: td.AddOp):
    typ = op.res.typ
    assert isinstance(typ, TensorType | UnrankedTensorType)
    typ = cast(td.AnyTensorTypeF64, typ)

    shape = tvd.TensorShapeOp(op.lhs)
    lhs = tvd.TensorDataOp(op.lhs)
    rhs = tvd.TensorDataOp(op.rhs)
    sum = tvd.VectorAddOp(lhs, rhs)
    result = tvd.TensorMakeOp(shape, sum, typ)

    return result.results


@implicit_rewriter
def lower_tensor_mul(op: td.AddOp):
    typ = op.res.typ
    assert isinstance(typ, TensorType | UnrankedTensorType)
    typ = cast(td.AnyTensorTypeF64, typ)

    shape = tvd.TensorShapeOp(op.lhs)
    lhs = tvd.TensorDataOp(op.lhs)
    rhs = tvd.TensorDataOp(op.rhs)
    prod = tvd.VectorMulOp(lhs, rhs)
    result = tvd.TensorMakeOp(shape, prod, typ)

    return result.results


class LowerToy(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(lower_tensor_constant).rewrite_module(op)
        PatternRewriteWalker(lower_reshape).rewrite_module(op)
        PatternRewriteWalker(lower_tensor_add).rewrite_module(op)
        PatternRewriteWalker(lower_tensor_mul).rewrite_module(op)
        PatternRewriteWalker(lower_print_op).rewrite_module(op)
        dce(op)
