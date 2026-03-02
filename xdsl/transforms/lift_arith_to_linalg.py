from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class LiftAddfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddfOp, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_op(
                op, linalg.AddOp(op.operands, [op.lhs], [op.result.type])
            )


class LiftSubfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SubfOp, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_op(
                op, linalg.SubOp(op.operands, [op.lhs], [op.result.type])
            )


class LiftMulfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MulfOp, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_op(
                op, linalg.MulOp(op.operands, [op.lhs], [op.result.type])
            )


@dataclass(frozen=True)
class LiftArithToLinalg(ModulePass):
    """
    Pass that lifts arith ops to linalg in order to make use of destination-passing style and bufferization.
    """

    name = "lift-arith-to-linalg"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LiftAddfPass(),
                    LiftSubfPass(),
                    LiftMulfPass(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
