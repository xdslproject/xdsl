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
    RewritePattern,
    Worklist,
)
from xdsl.rewriter import InsertPoint


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
        #          | OpOperands    ConstantOp   IntAttr
        #          |  |  OpResult   |  IntegerAttr | int
        #          |  |        |    |     |       /  |
        #          v  v        v    v     v      v   v
        lhs: int = op.operands[0].owner.value.value.data  # pyright: ignore
        rhs: int = op.operands[1].owner.value.value.data  # pyright: ignore
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
        """Apply the pass.

        This is a manual inlining of the call stack invoked by:

        ```python
        pattern = ConstantFoldingIntegerAdditionPattern()
        PatternRewriteWalker(pattern).rewrite_module(op)
        ```
        """
        ### Input values and state
        region = op.body
        op_was_modified = True
        walker_worklist = Worklist()

        ### The function implementation
        while op_was_modified:
            ## Inline `walker._populate_worklist(region)`
            for sub_op in region.walk(reverse=True, region_first=True):
                walker_worklist.push(sub_op)

            ## Inline `walker._process_worklist(listener)`
            rewriter_has_done_action = False

            # Handle empty worklist
            rewrite_op = walker_worklist.pop()
            if rewrite_op is None:
                op_was_modified = False
                continue

            # Create a rewriter on the first operation
            rewriter = PatternRewriter(rewrite_op)
            # No custom listeners have any effect, as we are operating in the
            # non-recursive mode and no operations are removed in constant folding.
            # As a result of this, we elide `rewriter.extend_from_listener(listener)`.

            # do/while loop
            while True:
                # Reset the rewriter on `op`
                rewriter.has_done_action = False
                rewriter.current_operation = rewrite_op
                rewriter.insertion_point = InsertPoint.before(rewrite_op)

                # Apply the pattern on the operation
                # Inline `walker.pattern.match_and_rewrite(rewrite_op, rewriter)`
                if isinstance(rewrite_op, AddiOp):
                    lhs: int = rewrite_op.operands[0].owner.value.value.data  # pyright: ignore
                    rhs: int = rewrite_op.operands[1].owner.value.value.data  # pyright: ignore
                    folded_op = ConstantOp(
                        IntegerAttr(lhs + rhs, rewrite_op.result.type)  # pyright: ignore
                    )
                    rewriter.replace_matched_op(folded_op, [folded_op.results[0]])

                rewriter_has_done_action |= rewriter.has_done_action

                # If the worklist is empty, we are done
                rewrite_op = walker_worklist.pop()
                if rewrite_op is None:
                    op_was_modified = rewriter_has_done_action
                    break
