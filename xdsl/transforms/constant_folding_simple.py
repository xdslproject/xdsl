"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.ir import ErasedSSAValue, Operation, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
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
        #          | OpOperands    ConstantOp   IntAttr
        #          |  |  OpResult   |  IntegerAttr | int
        #          |  |        |    |     |       /  |
        #          v  v        v    v     v      v   v
        lhs: int = op.operands[0].owner.value.value.data  # pyright: ignore
        rhs: int = op.operands[1].owner.value.value.data  # pyright: ignore
        folded_op = ConstantOp.create(
            attributes={
                "value": IntegerAttr(lhs + rhs, op.result.type)  # pyright: ignore
            }
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

        The remaining function invocations are walking the region, interacting
        with the worklist, and creating the new constant operation.
        """
        ### Input values
        region = op.body

        ### The function implementation
        op_was_modified = True
        walker_worklist = []  # Changed from `Worklist()`
        while op_was_modified:
            ## Inline `walker._populate_worklist(region)`
            # Elide `for sub_op in region.walk(reverse=True, region_first=True):`
            for sub_block in reversed(region.blocks):
                for sub_op in reversed(sub_block.ops):
                    walker_worklist.append(sub_op)

            ## Inline `walker._process_worklist(listener)`
            rewriter_has_done_action = False

            # Handle empty worklist
            rewrite_op = walker_worklist.pop() if len(walker_worklist) else None
            if rewrite_op is None:
                op_was_modified = False
                continue

            # No custom listeners have any effect, as we are operating in the
            # non-recursive mode and no operations are removed in constant folding.
            # As a result of this, we elide `rewriter.extend_from_listener(listener)`.

            # do/while loop
            while True:
                # Reset the rewriter on `op`
                # rewriter.current_operation = rewrite_op
                # rewriter.insertion_point = InsertPoint.before(rewrite_op)

                # Apply the pattern on the operation
                ## Inline `walker.pattern.match_and_rewrite(rewrite_op, rewriter)`
                if isinstance(rewrite_op, AddiOp):
                    lhs: int = rewrite_op.operands[0].owner.value.value.data  # pyright: ignore
                    rhs: int = rewrite_op.operands[1].owner.value.value.data  # pyright: ignore
                    folded_op = ConstantOp.create(
                        attributes={
                            "value": IntegerAttr(lhs + rhs, op.result.type)  # pyright: ignore
                        }
                    )
                    # ============================ #
                    ## Inline `rewriter.replace_matched_op(folded_op, [folded_op.results[0]])`
                    ## Inline `rewriter.replace_op(...)`
                    old_op = rewrite_op
                    new_results = [folded_op.results[0]]
                    rewriter_has_done_action = True

                    # First, insert the new operations before the matched operation
                    ## Inline `rewriter.insert_op((folded_op,), InsertPoint.before(old_op))`
                    ## There are no callbacks, so can elide `rewriter.handle_operation_insertion(op_)`
                    # ---------------------------- #
                    ## Inline `old_op.parent.insert_ops_before((folded_op,), old_op)`
                    ## Inline `old_op.parent.insert_op_before(folded_op, old_op)`
                    folded_op.parent = old_op.parent
                    prev_op = old_op.prev_op
                    ## Inline `old_op._insert_prev_op(folded_op)`
                    if old_op._prev_op is not None:  # pyright: ignore[reportPrivateUsage]
                        # update prev node
                        old_op._prev_op._next_op = folded_op  # pyright: ignore[reportPrivateUsage]
                    # set next and previous on new node
                    folded_op._prev_op = old_op._prev_op  # pyright: ignore[reportPrivateUsage]
                    folded_op._next_op = old_op  # pyright: ignore[reportPrivateUsage]
                    # update self
                    old_op._prev_op = folded_op  # pyright: ignore[reportPrivateUsage]
                    if prev_op is None:
                        # No `prev_op`, means `next_op` is the first op in the block.
                        old_op.parent._first_op = folded_op  # pyright: ignore[reportOptionalMemberAccess,reportPrivateUsage]
                    # ---------------------------- #

                    # Then, replace the results with new ones
                    ## There are no callbacks, so can elide `rewriter.handle_operation_replacement(op_)`
                    for old_result, new_result in zip(
                        old_op.results, new_results, strict=True
                    ):
                        ## Inline `rewriter._replace_all_uses_with(old_result, new_result, safe_erase=True)`
                        ## There are no callbacks, so can elide `self.handle_operation_modification(use.operation)`
                        ## Inline `old_result.replace_by(new_result)`
                        for use in old_result.uses.copy():
                            ## Inline `use.operation.operands.__setitem__(...)`
                            operands = use.operation._operands  # pyright: ignore[reportPrivateUsage]
                            operands[use.index].remove_use(
                                Use(use.operation, use.index)
                            )
                            new_result.add_use(Use(use.operation, use.index))
                            new_operands = (
                                *operands[: use.index],
                                new_result,
                                *operands[use.index + 1 :],
                            )
                            use.operation._operands = new_operands  # pyright: ignore[reportPrivateUsage]
                        new_result.name_hint = old_result.name_hint

                    # Then, erase the original operation
                    ## Inline `rewriter.erase_op(old_op, safe_erase=True)`
                    ## There are no callbacks, so can elide `rewriter.handle_operation_removal(old_op)`
                    ## Inline `Rewriter.erase_op(old_op, safe_erase=True)`
                    ## Inline `old_op.parent.erase_op(old_op, safe_erase=True)`
                    # ---------------------------- #
                    ## Inline `old_op = old_op.parent.detach_op(old_op)`
                    old_op.parent = None
                    prev_op = old_op.prev_op
                    next_op = old_op.next_op
                    if prev_op is not None:
                        # detach op from linked list
                        prev_op._next_op = next_op  # pyright: ignore[reportPrivateUsage]
                        # detach linked list from op
                        old_op._prev_op = None  # pyright: ignore[reportPrivateUsage]
                    else:
                        # reattach linked list if op is first op this block
                        old_op.parent._first_op = next_op  # pyright: ignore[reportAttributeAccessIssue]

                    if next_op is not None:
                        # detach op from linked list
                        next_op._prev_op = prev_op  # pyright: ignore[reportPrivateUsage]
                        # detach linked list from op
                        old_op._next_op = None  # pyright: ignore[reportPrivateUsage]
                    else:
                        # reattach linked list if op is last op in this block
                        old_op.parent._last_op = prev_op  # pyright: ignore[reportAttributeAccessIssue]
                    # ---------------------------- #
                    ## Inline `old_op.erase(safe_erase=True)`
                    ## Inline `old_op.drop_all_references()`
                    old_op.parent = None
                    for idx, operand in enumerate(old_op._operands):
                        ## Inline `operand.remove_use(Use(old_op, idx))`
                        operand.uses.remove(Use(old_op, idx))
                    ## This application has no regions, so no recursive drops

                    for result in old_op.results:
                        ## Inline `result.erase(safe_erase=True)`
                        ## Inline `result.replace_by(ErasedSSAValue(result.type, result))`
                        replace_value = ErasedSSAValue(result.type, result)
                        ## Newly constructed `ErasedSSAValue`s have no uses!
                        # for use in result.uses.copy():
                        #     use.operation.operands[use.index] = replace_value
                        # carry over name if possible
                        if replace_value.name_hint is None:
                            replace_value.name_hint = result.name_hint
                    # ============================ #

                # If the worklist is empty, we are done
                rewrite_op = walker_worklist.pop() if len(walker_worklist) else None
                if rewrite_op is None:
                    op_was_modified = rewriter_has_done_action
                    break
