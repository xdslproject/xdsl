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
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy as td
from ..dialects import vector as tvd

f64 = Float64Type()


class LowerTensorConstantOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ConstantOp, rewriter: PatternRewriter):
        typ = op.value.type

        assert isinstance(typ, TensorType), "Toy constants always have rank information"
        typ = cast(td.AnyTensorTypeF64, typ)

        shape = DenseIntOrFPElementsAttr.vector_from_list(op.get_shape(), i32)
        data = DenseIntOrFPElementsAttr.vector_from_list(op.get_data(), f64)

        shape_vector = tvd.VectorConstantOp(shape, "tensor_shape")
        data_vector = tvd.VectorConstantOp(data, "tensor_data")
        tensor = tvd.TensorMakeOp(shape_vector, data_vector, typ)

        rewriter.replace_matched_op([shape_vector, data_vector, tensor])


class LowerPrintOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.PrintOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            llvm.CallIntrinsicOp("tensor.print", (op.input,), ())
        )


class LowerReshapeOp(RewritePattern):
    def shape_data(self, shape: list[int]) -> list[int]:
        rank = len(shape)
        encoded_ints = [rank, *shape]
        return encoded_ints

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ReshapeOp, rewriter: PatternRewriter):
        typ = op.res.typ
        assert isinstance(typ, TensorType)
        typ = cast(td.TensorTypeF64, typ)
        shape = DenseIntOrFPElementsAttr.vector_from_list(typ.get_shape(), i32)

        new_shape = tvd.VectorConstantOp(shape, "tensor_new_shape")
        old_data = tvd.TensorDataOp(op.arg)
        make_tensor = tvd.TensorMakeOp(new_shape, old_data, typ)

        rewriter.replace_matched_op(
            [
                new_shape,
                old_data,
                make_tensor,
            ]
        )


class LowerTensorAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.AddOp, rewriter: PatternRewriter):
        typ = op.res.typ
        assert isinstance(typ, TensorType | UnrankedTensorType)
        typ = cast(td.AnyTensorTypeF64, typ)

        shape = tvd.TensorShapeOp(op.lhs)
        lhs = tvd.TensorDataOp(op.lhs)
        rhs = tvd.TensorDataOp(op.rhs)
        sum = tvd.VectorAddOp(lhs, rhs)
        result = tvd.TensorMakeOp(shape, sum, typ)

        rewriter.replace_matched_op([shape, lhs, rhs, sum, result])


class LowerTensorMulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.MulOp, rewriter: PatternRewriter):
        typ = op.res.typ
        assert isinstance(typ, TensorType | UnrankedTensorType)
        typ = cast(td.AnyTensorTypeF64, typ)

        shape = tvd.TensorShapeOp(op.lhs)
        lhs = tvd.TensorDataOp(op.lhs)
        rhs = tvd.TensorDataOp(op.rhs)
        prod = tvd.VectorMulOp(lhs, rhs)
        result = tvd.TensorMakeOp(shape, prod, typ)

        rewriter.replace_matched_op([shape, lhs, rhs, prod, result])


class LowerToy(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerTensorConstantOp()).rewrite_module(op)
        PatternRewriteWalker(LowerReshapeOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorAddOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorMulOp()).rewrite_module(op)
        PatternRewriteWalker(LowerPrintOp()).rewrite_module(op)
        dce(op)
