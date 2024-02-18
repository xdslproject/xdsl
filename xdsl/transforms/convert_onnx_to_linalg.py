from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, linalg, onnx, tensor
from xdsl.dialects.builtin import AffineMapAttr, FloatAttr, ModuleOp, TensorType, f64
from xdsl.ir import Block, MLContext, Region
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class AddOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, add: onnx.Add, rewriter: PatternRewriter, /):
        lhs_type = add.lhs.type
        rhs_type = add.rhs.type
        if isinstance(lhs_type, TensorType) and isinstance(rhs_type, TensorType):
            lhs_shape = lhs_type.get_shape()
            rhs_shape = rhs_type.get_shape()

            if 1 in lhs_shape or 1 in rhs_shape:
                raise NotImplementedError()

        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), add.res.type),
                linalg.AddOp((add.lhs, add.rhs), (empty.tensor,), res=(add.res.type,)),
            )
        )


@dataclass
class ReluOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, relu: onnx.Relu, rewriter: PatternRewriter, /):
        body = Region(Block(arg_types=(f64, f64)))
        affine_map = AffineMapAttr(AffineMap.from_callable(lambda d0, d1: (d0, d1)))
        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), relu.res.type),
                zero := arith.Constant(FloatAttr(0, f64)),
                linalg.Generic(
                    (relu.operand,),
                    (empty.tensor,),
                    body,
                    (
                        affine_map,
                        affine_map,
                    ),
                    (
                        linalg.IteratorTypeAttr.parallel(),
                        linalg.IteratorTypeAttr.parallel(),
                    ),
                    (relu.res.type,),
                ),
            )
        )
        with ImplicitBuilder(body) as (a, b):
            max_op = arith.Maximumf(a, zero.result)
            linalg.YieldOp(max_op.result)


@dataclass
class ConvertOnnxToLinalgPass(ModulePass):
    name = "convert-onnx-to-linalg"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                    ReluOpLowering(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
