from dataclasses import dataclass
from typing import cast

from xdsl.dialects import linalg, onnx, tensor
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Attribute, MLContext
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
        lhs_type = cast(TensorType[Attribute], add.lhs.type)
        rhs_type = cast(TensorType[Attribute], add.rhs.type)

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
class ConvertOnnxToLinalgPass(ModulePass):
    name = "convert-onnx-to-linalg"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
