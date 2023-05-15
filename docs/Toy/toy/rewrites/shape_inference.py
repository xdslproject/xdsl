from xdsl.ir import MLContext, Operation
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)
from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy


class InferShapes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        infer_shape_trait = op.get_trait(toy.ToyShapeInferenceTrait)

        if infer_shape_trait is None:
            return

        infer_shape_trait.infer_shape(op)


class RemoveCastOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.CastOp, rewriter: PatternRewriter):
        assert isinstance(op.arg.typ, toy.TensorType)
        assert isinstance(op.res.typ, toy.TensorType)
        assert op.arg.typ.get_shape() == op.res.typ.get_shape()
        rewriter.replace_matched_op([], new_results=op.operands)


class ShapeInferencePass(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(InferShapes()).rewrite_module(op)
        PatternRewriteWalker(RemoveCastOps()).rewrite_module(op)
        dce(op)
