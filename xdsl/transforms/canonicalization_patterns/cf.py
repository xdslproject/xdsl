from collections.abc import Sequence

from xdsl.dialects import arith, cf
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Block, BlockArgument, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


class AssertTrue(RewritePattern):
    """Erase assertion if argument is constant true."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.Assert, rewriter: PatternRewriter):
        owner = op.arg.owner

        if not isinstance(owner, arith.Constant):
            return

        value = owner.value

        if not isinstance(value, IntegerAttr):
            return

        if value.value.data != 1:
            return

        rewriter.replace_matched_op([])


class SimplifyBrToBlockWithSinglePred(RewritePattern):
    """
    Simplify a branch to a block that has a single predecessor. This effectively
    merges the two blocks.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.Branch, rewriter: PatternRewriter):
        succ = op.successor
        parent = op.parent_block()
        if parent is None:
            return

        # Check that the successor block has a single predecessor
        if succ == parent or len(succ.predecessors()) != 1:
            return

        br_operands = op.operands
        rewriter.erase_matched_op()
        rewriter.inline_block(succ, InsertPoint.at_end(parent), br_operands)


def collapse_branch(
    successor: Block, successor_operands: Sequence[SSAValue]
) -> tuple[Block, Sequence[SSAValue]] | None:
    """
    Given a successor, try to collapse it to a new destination if it only
    contains a passthrough unconditional branch. If the successor is
    collapsable, `successor` and `successorOperands` are updated to reference
    the new destination and values. `argStorage` is used as storage if operands
    to the collapsed successor need to be remapped. It must outlive uses of
    successorOperands.
    """

    # Check that successor only contains branch
    if len(successor.ops) != 1:
        return

    branch = successor.ops.first
    # Check that the terminator is an unconditional branch
    if not isinstance(branch, cf.Branch):
        return

    # Check that the arguments are only used within the terminator
    for argument in successor.args:
        for user in argument.uses:
            if user.operation != branch:
                return

    # Don't try to collapse branches to infinite loops.
    if branch.successor == successor:
        return

    # Remap operands
    operands = branch.operands

    new_operands = tuple(
        successor_operands[operand.index]
        if isinstance(operand, BlockArgument) and operand.owner is successor
        else operand
        for operand in operands
    )

    return (branch.successor, new_operands)


class SimplifyPassThroughBr(RewritePattern):
    """
      br ^bb1
    ^bb1
      br ^bbN(...)

     -> br ^bbN(...)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.Branch, rewriter: PatternRewriter):
        # Check the successor doesn't point back to the current block
        parent = op.parent_block()
        if parent is None or op.successor == parent:
            return

        ret = collapse_branch(op.successor, op.arguments)
        if ret is None:
            return
        (block, args) = ret

        rewriter.replace_matched_op(cf.Branch(block, *args))
