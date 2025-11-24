from collections.abc import Callable, Sequence
from typing import cast

from xdsl.dialects import arith, cf
from xdsl.dialects.builtin import (
    BoolAttr,
    DenseIntElementsAttr,
    IntegerAttr,
    VectorType,
)
from xdsl.ir import Block, BlockArgument, Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalization_patterns.utils import (
    const_evaluate_operand,
)


class AssertTrue(RewritePattern):
    """Erase assertion if argument is constant true."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.AssertOp, rewriter: PatternRewriter):
        owner = op.arg.owner

        if not isinstance(owner, arith.ConstantOp):
            return

        value = owner.value

        if not isinstance(value, IntegerAttr):
            return

        if not value.value.data:
            return

        rewriter.replace_op(op, [])


class SimplifyBrToBlockWithSinglePred(RewritePattern):
    """
    Simplify a branch to a block that has a single predecessor. This effectively
    merges the two blocks.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.BranchOp, rewriter: PatternRewriter):
        succ = op.successor
        parent = op.parent_block()
        if parent is None:
            return

        # Check that the successor block has a single predecessor
        if succ == parent or len(succ.predecessors()) != 1:
            return

        br_operands = op.operands
        rewriter.erase_op(op)
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
    if not isinstance(branch, cf.BranchOp):
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
    def match_and_rewrite(self, op: cf.BranchOp, rewriter: PatternRewriter):
        # Check the successor doesn't point back to the current block
        parent = op.parent_block()
        if parent is None or op.successor == parent:
            return

        ret = collapse_branch(op.successor, op.arguments)
        if ret is None:
            return
        (block, args) = ret

        rewriter.replace_op(op, cf.BranchOp(block, *args))


class SimplifyConstCondBranchPred(RewritePattern):
    """
    cf.cond_br true, ^bb1, ^bb2
     -> br ^bb1
    cf.cond_br false, ^bb1, ^bb2
     -> br ^bb2
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranchOp, rewriter: PatternRewriter):
        # Check if cond operand is constant
        cond = const_evaluate_operand(op.cond)

        if cond is None:
            return

        if cond:
            rewriter.replace_op(op, cf.BranchOp(op.then_block, *op.then_arguments))
        else:
            rewriter.replace_op(op, cf.BranchOp(op.else_block, *op.else_arguments))


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
    def match_and_rewrite(self, op: cf.ConditionalBranchOp, rewriter: PatternRewriter):
        # Try to collapse both branches
        collapsed_then = collapse_branch(op.then_block, op.then_arguments)
        collapsed_else = collapse_branch(op.else_block, op.else_arguments)

        # If neither collapsed then we return
        if collapsed_then is None and collapsed_else is None:
            return

        (new_then, new_then_args) = collapsed_then or (op.then_block, op.then_arguments)

        (new_else, new_else_args) = collapsed_else or (op.else_block, op.else_arguments)

        rewriter.replace_op(
            op,
            cf.ConditionalBranchOp(
                op.cond, new_then, new_then_args, new_else, new_else_args
            ),
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
        cond_br: cf.ConditionalBranchOp,
    ) -> SSAValue:
        if op1 == op2:
            return op1
        select = arith.SelectOp(cond_br.cond, op1, op2)
        rewriter.insert_op(select, InsertPoint.before(cond_br))
        return select.result

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranchOp, rewriter: PatternRewriter):
        # Check that the true and false destinations are the same
        if op.then_block != op.else_block:
            return

        merged_operands = tuple(
            self._merge_operand(op1, op2, rewriter, op)
            for (op1, op2) in zip(op.then_arguments, op.else_arguments, strict=True)
        )

        rewriter.replace_op(op, cf.BranchOp(op.then_block, *merged_operands))


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
    def match_and_rewrite(self, op: cf.ConditionalBranchOp, rewriter: PatternRewriter):
        if len(op.then_block.predecessors()) == 1:
            if any(
                use.operation.parent_block() is op.then_block for use in op.cond.uses
            ):
                const_true = arith.ConstantOp(BoolAttr.from_bool(True))
                rewriter.insert_op(const_true, InsertPoint.before(op))
                op.cond.replace_by_if(
                    const_true.result,
                    lambda use: use.operation.parent_block() is op.then_block,
                )
        if len(op.else_block.predecessors()) == 1:
            if any(
                use.operation.parent_block() is op.else_block for use in op.cond.uses
            ):
                const_false = arith.ConstantOp(BoolAttr.from_bool(False))
                rewriter.insert_op(const_false, InsertPoint.before(op))
                op.cond.replace_by_if(
                    const_false.result,
                    lambda use: use.operation.parent_block() is op.else_block,
                )


class SimplifySwitchWithOnlyDefault(RewritePattern):
    """
    switch %flag : i32, [
      default:  ^bb1
    ]
     -> br ^bb1
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.SwitchOp, rewriter: PatternRewriter):
        if not op.case_blocks:
            rewriter.replace_op(op, cf.BranchOp(op.default_block, *op.default_operands))


def drop_case_helper(
    rewriter: PatternRewriter,
    op: cf.SwitchOp,
    predicate: Callable[[IntegerAttr, Block, Sequence[Operation | SSAValue]], bool],
):
    case_values = op.case_values
    if case_values is None:
        return
    requires_change = False

    new_case_values: list[int] = []
    new_case_blocks: list[Block] = []
    new_case_operands: list[Sequence[Operation | SSAValue]] = []

    for switch_case, block, operands in zip(
        case_values.get_attrs(),
        op.case_blocks,
        op.case_operand,
        strict=True,
    ):
        int_switch_case = cast(IntegerAttr, switch_case)
        if predicate(int_switch_case, block, operands):
            requires_change = True
            continue
        new_case_values.append(cast(IntegerAttr, switch_case).value.data)
        new_case_blocks.append(block)
        new_case_operands.append(operands)

    if requires_change:
        rewriter.replace_op(
            op,
            cf.SwitchOp(
                op.flag,
                op.default_block,
                op.default_operands,
                DenseIntElementsAttr.from_list(
                    VectorType(case_values.get_element_type(), (len(new_case_values),)),
                    new_case_values,
                ),
                new_case_blocks,
                new_case_operands,
            ),
        )


class DropSwitchCasesThatMatchDefault(RewritePattern):
    """
    switch %flag : i32, [
      default: ^bb1,
      42: ^bb1,
      43: ^bb2
    ]
    ->
    switch %flag : i32, [
      default: ^bb1,
      43: ^bb2
    ]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.SwitchOp, rewriter: PatternRewriter):
        def predicate(
            switch_case: IntegerAttr,
            block: Block,
            operands: Sequence[Operation | SSAValue],
        ) -> bool:
            return block == op.default_block and operands == op.default_operands

        drop_case_helper(rewriter, op, predicate)


def fold_switch(switch: cf.SwitchOp, rewriter: PatternRewriter, flag: int):
    """
    Helper for folding a switch with a constant value.
    switch %c_42 : i32, [
      default: ^bb1 ,
      42: ^bb2,
      43: ^bb3
    ]
    -> br ^bb2
    """
    case_values = () if switch.case_values is None else switch.case_values.get_attrs()

    new_block, new_operands = next(
        (
            (block, operand)
            for (c, block, operand) in zip(
                case_values, switch.case_blocks, switch.case_operand, strict=True
            )
            if flag == c.value.data
        ),
        (switch.default_block, switch.default_operands),
    )

    rewriter.replace_op(switch, cf.BranchOp(new_block, *new_operands))


class SimplifyConstSwitchValue(RewritePattern):
    """
    switch %c_42 : i32, [
      default: ^bb1,
      42: ^bb2,
      43: ^bb3
    ]
    -> br ^bb2
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.SwitchOp, rewriter: PatternRewriter):
        if (flag := const_evaluate_operand(op.flag)) is not None:
            fold_switch(op, rewriter, flag)


class SimplifyPassThroughSwitch(RewritePattern):
    """
    switch %c_42 : i32, [
      default: ^bb1,
      42: ^bb2,
    ]
    ^bb2:
      br ^bb3
    ->
    switch %c_42 : i32, [
      default: ^bb1,
      42: ^bb3,
    ]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.SwitchOp, rewriter: PatternRewriter):
        requires_change = False

        new_case_blocks: list[Block] = []
        new_case_operands: list[Sequence[Operation | SSAValue]] = []

        for block, operands in zip(op.case_blocks, op.case_operand, strict=True):
            collapsed = collapse_branch(block, operands)
            requires_change |= collapsed is not None
            (new_block, new_operands) = collapsed or (block, operands)
            new_case_blocks.append(new_block)
            new_case_operands.append(new_operands)

        collapsed = collapse_branch(op.default_block, op.default_operands)

        requires_change |= collapsed is not None

        (default_block, default_operands) = collapsed or (
            op.default_block,
            op.default_operands,
        )

        if requires_change:
            rewriter.replace_op(
                op,
                cf.SwitchOp(
                    op.flag,
                    default_block,
                    default_operands,
                    op.case_values,
                    new_case_blocks,
                    new_case_operands,
                ),
            )


class SimplifySwitchFromSwitchOnSameCondition(RewritePattern):
    """
    switch %flag : i32, [
      default: ^bb1,
      42: ^bb2,
    ]
    ^bb2:
      switch %flag : i32, [
        default: ^bb3,
        42: ^bb4
      ]
    ->
    switch %flag : i32, [
      default: ^bb1,
      42: ^bb2,
    ]
    ^bb2:
      br ^bb4

     and

    switch %flag : i32, [
      default: ^bb1,
      42: ^bb2,
    ]
    ^bb2:
      switch %flag : i32, [
        default: ^bb3,
        43: ^bb4
      ]
    ->
    switch %flag : i32, [
      default: ^bb1,
      42: ^bb2,
    ]
    ^bb2:
      br ^bb3

    and

    switch %flag : i32, [
      default: ^bb1,
      42: ^bb2
    ]
    ^bb1:
      switch %flag : i32, [
        default: ^bb3,
        42: ^bb4,
        43: ^bb5
      ]
    ->
    switch %flag : i32, [
      default: ^bb1,
      42: ^bb2,
    ]
    ^bb1:
      switch %flag : i32, [
        default: ^bb3,
        43: ^bb5
      ]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.SwitchOp, rewriter: PatternRewriter):
        block = op.parent_block()
        if block is None:
            return
        if (pred := block.get_unique_use()) is None:
            return
        switch = pred.operation
        if not isinstance(switch, cf.SwitchOp):
            return

        if switch.flag != op.flag:
            return

        case_values = switch.case_values
        if case_values is None:
            return

        if pred.index != 0:
            fold_switch(
                op,
                rewriter,
                case_values.get_values()[pred.index - 1],
            )
        else:

            def predicate(
                switch_case: IntegerAttr,
                block: Block,
                operands: Sequence[Operation | SSAValue],
            ) -> bool:
                return switch_case in case_values.get_attrs()

            drop_case_helper(rewriter, op, predicate)
