"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntAttr, IntegerAttr, ModuleOp
from xdsl.ir import ErasedSSAValue, Operation, OpResult
from xdsl.irdl import SSAValues
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import ConstantLike


@dataclass
class TestConstantFoldingIntegerAdditionPattern(RewritePattern):
    """Rewrite pattern that constant folds integer types."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        # Only rewrite integer add operations
        if not isinstance(op, AddiOp):
            return

        # Ensure both operands are constants
        lhs_op: ConstantOp = op.operands[0].op  # pyright: ignore
        rhs_op: ConstantOp = op.operands[1].op  # pyright: ignore
        assert lhs_op.has_trait(ConstantLike)  # pyright: ignore
        assert rhs_op.has_trait(ConstantLike)  # pyright: ignore

        # Calculate the result of the addition
        lhs: int = lhs_op.value.value.data  # pyright: ignore
        rhs: int = rhs_op.value.value.data  # pyright: ignore
        folded_op = ConstantOp(
            IntegerAttr(lhs + rhs, op.result.type)  # pyright: ignore[reportCallIssue, reportArgumentType]
        )

        # Rewrite with the calculated result
        rewriter.replace_op(op, folded_op, [folded_op.results[0]])


class TestConstantFoldingPass(ModulePass):
    """
    A pass that applies applies simple constant folding.
    """

    name = "test-constant-folding"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the pass."""
        pattern = TestConstantFoldingIntegerAdditionPattern()
        PatternRewriteWalker(pattern).rewrite_module(op)


class TestSpecialisedConstantFoldingPass(ModulePass):
    """
    A pass that applies applies simple constant folding.
    """

    name = "test-specialised-constant-folding"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the pass.

        This is a full manual inlining of the call stack invoked by:

        ```python
        pattern = ConstantFoldingIntegerAdditionPattern()
        PatternRewriteWalker(pattern).rewrite_module(op)
        ```
        """
        region = op.regions[0]
        op_was_modified = True
        worklist: list[Operation] = []  # Changed from `Worklist()`
        while op_was_modified:
            # Avoid constructing iterators like `for sub_block in reversed(region.blocks)`
            sub_block = region.last_block
            while sub_block is not None:
                sub_op = sub_block.last_op
                while sub_op is not None:
                    worklist.append(sub_op)
                    sub_op = sub_op.prev_op
                sub_block = sub_block.prev_block

            # Handle empty worklist
            rewrite_op = worklist.pop() if len(worklist) else None
            if rewrite_op is None:
                op_was_modified = False
                continue

            # No custom listeners have any effect, as we are operating in the
            # non-recursive mode and no operations are removed in constant folding.
            # As a result of this, we elide `rewriter.extend_from_listener(listener)`.
            while True:
                # Elide exception handling
                rewriter_has_done_action = False
                if isinstance(rewrite_op, AddiOp):
                    lhs_op: OpResult = rewrite_op.operands[0].op  # pyright: ignore
                    rhs_op: OpResult = rewrite_op.operands[1].op  # pyright: ignore

                    constant_like = ConstantLike
                    has_trait = False
                    for t in lhs_op.traits._traits:  # pyright: ignore
                        if isinstance(t, constant_like):
                            has_trait = True
                            break
                    assert has_trait
                    has_trait = False
                    for t in rhs_op.traits._traits:  # pyright: ignore
                        if isinstance(t, constant_like):
                            has_trait = True
                            break
                    assert has_trait

                    lhs: int = lhs_op.value.value.data  # pyright: ignore
                    rhs: int = rhs_op.value.value.data  # pyright: ignore

                    result_type = rewrite_op.results[0].type
                    ## Inline `IntegerAttr(lhs + rhs, result_type)`
                    int_attr = IntAttr.__new__(IntAttr)
                    object.__setattr__(int_attr, "data", lhs + rhs)
                    integer_attr = IntegerAttr.__new__(IntegerAttr)
                    ## Inline `ParametrizedAttribute.__init__(integer_attr,[int_attr, result_type])`
                    object.__setattr__(integer_attr, "value", int_attr)
                    object.__setattr__(integer_attr, "type", result_type)
                    folded_op = ConstantOp.__new__(ConstantOp)
                    folded_op._operands = SSAValues()  # pyright: ignore[reportPrivateUsage]
                    folded_op.results = SSAValues(
                        (OpResult(result_type, folded_op, 0),)
                    )
                    folded_op.properties = {"value": integer_attr}
                    folded_op.attributes = {}
                    folded_op._successors = tuple()  # pyright: ignore[reportPrivateUsage]
                    folded_op.regions = tuple()

                    old_op, new_op, new_result = (
                        rewrite_op,
                        folded_op,
                        folded_op.results[0],
                    )

                    # First, insert the new operations before the matched operation
                    new_op.parent = old_op.parent

                    prev_op = old_op.prev_op
                    if old_op._prev_op is not None:  # pyright: ignore[reportPrivateUsage]
                        # update prev node
                        old_op._prev_op._next_op = new_op  # pyright: ignore[reportPrivateUsage]
                    # set next and previous on new node
                    new_op._prev_op = old_op._prev_op  # pyright: ignore[reportPrivateUsage]
                    new_op._next_op = old_op  # pyright: ignore[reportPrivateUsage]
                    # update self
                    old_op._prev_op = new_op  # pyright: ignore[reportPrivateUsage]

                    if prev_op is None:
                        # No `prev_op`, means `next_op` is the first op in the block.
                        old_op.parent._first_op = new_op  # pyright: ignore[reportOptionalMemberAccess,reportPrivateUsage]

                    ## There are no callbacks, so can elide `rewriter.handle_operation_replacement(op_)`

                    # Then, replace the results with new ones
                    ## We know there is only one result, so can elide the loop
                    old_result = old_op.results[0]
                    ## There are no callbacks, so can elide `self.handle_operation_modification(use.operation)`
                    for use in tuple(old_result.uses):
                        ## Inline `use.operation.operands.__setitem__(...)`
                        operands = use.operation._operands  # pyright: ignore[reportPrivateUsage]
                        ## Inline `operands[use.index].remove_use(Use(use.operation, use.index))`
                        old_use = use.operation._operand_uses[use.index]  # pyright: ignore[reportPrivateUsage]
                        prev_use = old_use._prev_use  # pyright: ignore[reportPrivateUsage]
                        next_use = old_use._next_use  # pyright: ignore[reportPrivateUsage]
                        if prev_use is not None:
                            prev_use._next_use = next_use  # pyright: ignore[reportPrivateUsage]
                        if next_use is not None:
                            next_use._prev_use = prev_use  # pyright: ignore[reportPrivateUsage]

                        if prev_use is None:
                            operands[use.index].first_use = next_use
                        ## Inline `new_result.add_use(Use(use.operation, use.index))`
                        first_use = new_result.first_use
                        use._next_use = first_use  # pyright: ignore[reportPrivateUsage]
                        use._prev_use = None  # pyright: ignore[reportPrivateUsage]
                        if first_use is not None:
                            first_use._prev_use = use  # pyright: ignore[reportPrivateUsage]
                        new_result.first_use = use
                        new_operands = SSAValues(
                            (
                                *operands[: use.index],
                                new_result,
                                *operands[use.index + 1 :],
                            )
                        )
                        use.operation._operands = new_operands  # pyright: ignore[reportPrivateUsage]
                    new_result.name_hint = old_result.name_hint

                    # Elide "preserv[ing] name hints for ops with multiple results",
                    # since done already in `SSAValue.replace_by`

                    # Elide "add[ing] name hints for existing ops, only if there is a
                    # single new result", since done already in `SSAValue.replace_by`

                    ## There are no callbacks, so can elide `rewriter.handle_operation_removal(old_op)`
                    block = old_op.parent
                    if block is not None:
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
                            block._first_op = next_op  # pyright: ignore[reportPrivateUsage]

                        if next_op is not None:
                            # detach op from linked list
                            next_op._prev_op = prev_op  # pyright: ignore[reportPrivateUsage]
                            # detach linked list from op
                            old_op._next_op = None  # pyright: ignore[reportPrivateUsage]
                        else:
                            # reattach linked list if op is last op in this block
                            block._last_op = prev_op  # pyright: ignore[reportPrivateUsage]

                    old_op.parent = None
                    for operand, use in zip(old_op._operands, old_op._operand_uses):  # pyright: ignore[reportPrivateUsage]
                        ## Inline `operand.remove_use(Use(old_op, idx))`
                        prev_use = use._prev_use  # pyright: ignore[reportPrivateUsage]
                        next_use = use._next_use  # pyright: ignore[reportPrivateUsage]
                        if prev_use is not None:
                            prev_use._next_use = next_use  # pyright: ignore[reportPrivateUsage]
                        if next_use is not None:
                            next_use._prev_use = prev_use  # pyright: ignore[reportPrivateUsage]

                        if prev_use is None:
                            operand.first_use = next_use
                    ## This application has no regions, so no recursive drops

                    for result in old_op.results:
                        ## Inline `result.replace_by(ErasedSSAValue(result.type, result))`
                        replace_value = ErasedSSAValue(result.type, result)
                        ## Newly constructed `ErasedSSAValue`s have no uses, so can
                        ## elide copying its uses
                        # carry over name if possible
                        if replace_value.name_hint is None:
                            replace_value.name_hint = result.name_hint
                    rewriter_has_done_action = True

                # If the worklist is empty, we are done
                rewrite_op = worklist.pop() if len(worklist) else None
                if rewrite_op is None:
                    op_was_modified = rewriter_has_done_action
                    break
