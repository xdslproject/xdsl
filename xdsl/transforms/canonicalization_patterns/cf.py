from collections.abc import Sequence

from xdsl.dialects import arith, cf
from xdsl.dialects.builtin import BoolAttr, IntegerAttr
from xdsl.ir import Block, BlockArgument, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand


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


class SimplifyConstCondBranchPred(RewritePattern):
    """
    cf.cond_br true, ^bb1, ^bb2
     -> br ^bb1
    cf.cond_br false, ^bb1, ^bb2
     -> br ^bb2
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranch, rewriter: PatternRewriter):
        # Check if cond operand is constant
        cond = const_evaluate_operand(op.cond)

        if cond == 1:
            rewriter.replace_matched_op(cf.Branch(op.then_block, *op.then_arguments))
        elif cond == 0:
            rewriter.replace_matched_op(cf.Branch(op.else_block, *op.else_arguments))


class SimplifyPassThroughCondBranch(RewritePattern):
    """
      cf.cond_br %cond, ^bb1, ^bb2
    ^bb1
      br ^bbN(...)
    ^bb2
      br ^bbK(...)

     -> cf.cond_br %cond, ^bbN(...), ^bbK(...)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranch, rewriter: PatternRewriter):
        # Try to collapse both branches
        collapsed_then = collapse_branch(op.then_block, op.then_arguments)
        collapsed_else = collapse_branch(op.else_block, op.else_arguments)

        # If neither collapsed then we return
        if collapsed_then is None and collapsed_else is None:
            return

        (new_then, new_then_args) = collapsed_then or (op.then_block, op.then_arguments)

        (new_else, new_else_args) = collapsed_else or (op.else_block, op.else_arguments)

        rewriter.replace_matched_op(
            cf.ConditionalBranch(
                op.cond, new_then, new_then_args, new_else, new_else_args
            )
        )


class SimplifyCondBranchIdenticalSuccessors(RewritePattern):
    """
    cf.cond_br %cond, ^bb1(A, ..., N), ^bb1(A, ..., N)
     -> br ^bb1(A, ..., N)

    cf.cond_br %cond, ^bb1(A), ^bb1(B)
     -> %select = arith.select %cond, A, B
        br ^bb1(%select)
    """

    @staticmethod
    def _merge_operand(
        op1: SSAValue,
        op2: SSAValue,
        rewriter: PatternRewriter,
        cond_br: cf.ConditionalBranch,
    ) -> SSAValue:
        if op1 == op2:
            return op1
        select = arith.Select(cond_br.cond, op1, op2)
        rewriter.insert_op(select, InsertPoint.before(cond_br))
        return select.result

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranch, rewriter: PatternRewriter):
        # Check that the true and false destinations are the same
        if op.then_block != op.else_block:
            return

        merged_operands = tuple(
            self._merge_operand(op1, op2, rewriter, op)
            for (op1, op2) in zip(op.then_arguments, op.else_arguments, strict=True)
        )

        rewriter.replace_matched_op(cf.Branch(op.then_block, *merged_operands))


class CondBranchTruthPropagation(RewritePattern):
    """
      cf.cond_br %arg0, ^trueB, ^falseB

    ^trueB:
      "test.consumer1"(%arg0) : (i1) -> ()
       ...

    ^falseB:
      "test.consumer2"(%arg0) : (i1) -> ()
      ...

    ->

      cf.cond_br %arg0, ^trueB, ^falseB
    ^trueB:
      "test.consumer1"(%true) : (i1) -> ()
      ...

    ^falseB:
      "test.consumer2"(%false) : (i1) -> ()
      ...
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranch, rewriter: PatternRewriter):
        if len(op.then_block.predecessors()) == 1:
            if any(
                use.operation.parent_block() is op.then_block for use in op.cond.uses
            ):
                const_true = arith.Constant(BoolAttr.from_bool(True))
                rewriter.insert_op(const_true, InsertPoint.before(op))
                op.cond.replace_by_if(
                    const_true.result,
                    lambda use: use.operation.parent_block() is op.then_block,
                )
        if len(op.else_block.predecessors()) == 1:
            if any(
                use.operation.parent_block() is op.else_block for use in op.cond.uses
            ):
                const_false = arith.Constant(BoolAttr.from_bool(False))
                rewriter.insert_op(const_false, InsertPoint.before(op))
                op.cond.replace_by_if(
                    const_false.result,
                    lambda use: use.operation.parent_block() is op.else_block,
                )
