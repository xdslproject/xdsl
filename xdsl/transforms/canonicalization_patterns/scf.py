from collections.abc import Sequence

from xdsl.dialects import scf
from xdsl.ir import Operation, Region, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import ConstantLike
from xdsl.transforms.canonicalization_patterns.utils import (
    const_evaluate_operand,
)


class RehoistConstInLoops(RewritePattern):
    """
    Carry out const definitions from the loops.
    In the future this will probably be done by the pattern rewriter itself, like it's
    done in the MLIR's applyPatternsAndFoldGreedily.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        for child_op in op.body.ops:
            if child_op.has_trait(ConstantLike):
                # we only rehoist consts that are not embeded in another region inside the loop
                rewriter.insert_op(new_const := child_op.clone())
                rewriter.replace_op(child_op, (), new_const.results)


class SimplifyTrivialLoops(RewritePattern):
    """
    Rewriting pattern that erases loops that are known not to iterate, replaces
    single-iteration loops with their bodies, and removes empty loops that iterate at
    least once and only return values defined outside of the loop.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        # If the upper bound is the same as the lower bound, the loop does not iterate,
        # just remove it.
        if (lb := const_evaluate_operand(op.lb)) is None:
            return
        if (ub := const_evaluate_operand(op.ub)) is None:
            return

        if lb == ub:
            rewriter.replace_op(op, (), op.iter_args)
            return

        # If the loop is known to have 0 iterations, remove it.
        if (diff := ub - lb) <= 0:
            rewriter.replace_op(op, (), op.iter_args)
            return

        if (step := const_evaluate_operand(op.step)) is None:
            return

        # If the loop is known to have 1 iteration, inline its body and remove the loop.
        # TODO: handle signless values
        if step >= diff:
            block_args = (op.lb, *op.iter_args)
            replace_op_with_region(
                rewriter,
                op,
                op.body,
                block_args,
            )

        # Now we are left with loops that have more than 1 iterations.
        # block = op.body.block
        # if isinstance(block.first_op, scf.Yield):
        #     return

        # If the loop is empty, iterates at least once, and only returns values defined
        # outside of the loop, remove it and replace it with yield values.
        # TODO: https://mlir.llvm.org/doxygen/Dialect_2SCF_2IR_2SCF_8cpp_source.html


def replace_op_with_region(
    rewriter: PatternRewriter,
    op: Operation,
    region: Region,
    args: Sequence[SSAValue] = (),
):
    """
    Replaces the given op with the contents of the given single-block region, using the
    operands of the block terminator to replace operation results.
    """

    block = region.block
    terminator = block.last_op
    assert terminator is not None
    rewriter.inline_block(block, InsertPoint.before(op), args)
    rewriter.replace_op(op, (), terminator.operands)
    rewriter.erase_op(terminator)
