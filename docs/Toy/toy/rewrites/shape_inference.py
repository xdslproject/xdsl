from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
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
        assert isinstance(op_arg_type := op.arg.type, toy.TensorType)
        assert isinstance(op_res_type := op.res.type, toy.TensorType)
        assert op_arg_type.get_shape() == op_res_type.get_shape()
        rewriter.replace_matched_op([], new_results=op.operands)


class ShapeInferencePass(ModulePass):
    name = "toy-infer-shapes"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(InferShapes()).rewrite_module(op)
        PatternRewriteWalker(RemoveCastOps()).rewrite_module(op)
        dce(op)
