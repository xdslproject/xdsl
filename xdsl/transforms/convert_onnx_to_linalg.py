from dataclasses import dataclass

from xdsl.dialects import linalg, onnx, tensor
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
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
        lhs_shape = add.lhs.type.get_shape()
        rhs_shape = add.rhs.type.get_shape()

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
