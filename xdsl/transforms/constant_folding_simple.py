"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp, SignlessIntegerBinaryOperation
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.ir import Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import ConstantLike


@dataclass
class ConstantFoldingIntegerAdditionPattern(RewritePattern):
    """Rewrite pattern that constant folds integer types."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        # Only rewrite operations for which `py_operation` is defined, hence
        # having the parent class `SignlessIntegerBinaryOperation`.
        if not isinstance(op, SignlessIntegerBinaryOperation):
            return

        # No need to rewrite operations that are already constant-like
        if op.has_trait(ConstantLike):
            return

        # Only rewrite operations where all the operands are integer constants
        for operand in op.operands:
            if not isinstance(operand, OpResult) or not operand.op.has_trait(
                ConstantLike
            ):
                return

        # Calculate the result of the addition
        folded_value = op.py_operation(
            # SignlessIntegerBinaryOperation->OpOperands->OpResult->ConstantOp
            #     ->IntegerAttr->IntAttr->int
            lhs=op.operands[0].op.value.value.data,  # pyright: ignore
            rhs=op.operands[1].op.value.value.data,  # pyright: ignore
        )
        folded_op = ConstantOp(
            IntegerAttr(folded_value, op.result.type)  # pyright: ignore
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
