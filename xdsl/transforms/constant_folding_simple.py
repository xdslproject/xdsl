"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)


@dataclass
class ConstantFoldingIntegerAdditionPattern(RewritePattern):
    """Rewrite pattern that constant folds integer types."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        # Only rewrite integer add operations
        if not isinstance(op, AddiOp):
            return

        # # Only rewrite operations where all the operands are integer constants
        # for operand in op.operands:
        #     assert isinstance(operand, OpResult)
        #     assert operand.op.has_trait(ConstantLike)

        # Calculate the result of the addition
        #
        #  SignlessIntegerBinaryOperation
        #          | OpOperands  ConstantOp   IntAttr
        #          |  |  OpResult | IntegerAttr | int
        #          |  |        |  |    |       /  |
        #          v  v        v  v    v      v   v
        lhs: int = op.operands[0].op.value.value.data  # pyright: ignore
        rhs: int = op.operands[1].op.value.value.data  # pyright: ignore
        folded_op = ConstantOp(
            IntegerAttr(lhs + rhs, op.result.type)  # pyright: ignore
        )

        # Rewrite with the calculated result
        rewriter.replace_matched_op(folded_op, [folded_op.results[0]])


class ConstantFoldingSimplePass(ModulePass):
    """
    A pass that applies applies simple constant folding.
    """

    name = "constant-folding-simple"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        pattern = ConstantFoldingIntegerAdditionPattern()
        PatternRewriteWalker(pattern).rewrite_module(op)
