"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntAttr, IntegerAttr, ModuleOp
from xdsl.ir import ErasedSSAValue, Operation, OpResult, Region, SSAValue, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import ConstantLike, OpTrait


@dataclass
class ConstantFoldingIntegerAdditionPattern(RewritePattern):
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
        rewriter.replace_matched_op(folded_op, [folded_op.results[0]])


class ConstantFoldingSimplePass(ModulePass):
    """
    A pass that applies applies simple constant folding.
    """

    name = "constant-folding-simple"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the pass."""
        # self.apply_pattern(ctx, op)
        # self.apply_split(ctx, op)
        self.apply_inlined(ctx, op)

    def apply_pattern(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the pass using the pattern rewriter."""
        pattern = ConstantFoldingIntegerAdditionPattern()
        PatternRewriteWalker(pattern).rewrite_module(op)

    def apply_split(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the pass.

        This is a manual inlining of the call stack invoked by:

        ```python
        pattern = ConstantFoldingIntegerAdditionPattern()
        PatternRewriteWalker(pattern).rewrite_module(op)
        ```
        """

        def drop_all_references(detached_op: Operation) -> None:
            """Drop all references to other operations."""
            detached_op.parent = None
            for idx, operand in enumerate(detached_op._operands):
                ## Inline `operand.remove_use(Use(detached_op, idx))`
                operand.uses.remove(Use(detached_op, idx))
            ## This application has no regions, so no recursive drops

        def op_erase(detached_op: Operation) -> None:
            """Erase the operation, and remove all its references to other operations."""
            drop_all_references(detached_op)
            for result in detached_op.results:
                ## Inline `result.replace_by(ErasedSSAValue(result.type, result))`
                replace_value = ErasedSSAValue(result.type, result)
                ## Newly constructed `ErasedSSAValue`s have no uses, so can
                ## elide copying its uses
                # carry over name if possible
                if replace_value.name_hint is None:
                    replace_value.name_hint = result.name_hint

        def detach_op(old_op: Operation) -> None:
            """Detach an operation from the block."""
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
            return old_op

        def erase_op(old_op: Operation) -> None:
            """Erase an operation from the block."""
            ## There are no callbacks, so can elide `rewriter.handle_operation_removal(old_op)`
            detached_op = detach_op(old_op)
            op_erase(detached_op)

        def replace_all_uses_with(old_result: OpResult, new_result: SSAValue) -> None:
            """Replace all uses of an SSA value with another SSA value."""
            ## There are no callbacks, so can elide `self.handle_operation_modification(use.operation)`
            for use in old_result.uses.copy():
                ## Inline `use.operation.operands.__setitem__(...)`
                operands = use.operation._operands  # pyright: ignore[reportPrivateUsage]
                ## Inline `operands[use.index].remove_use(Use(use.operation, use.index))`
                operands[use.index].uses.remove(use)
                ## Inline `new_result.add_use(Use(use.operation, use.index))`
                new_result.uses.add(use)
                new_operands = (
                    *operands[: use.index],
                    new_result,
                    *operands[use.index + 1 :],
                )
                use.operation._operands = new_operands  # pyright: ignore[reportPrivateUsage]
            new_result.name_hint = old_result.name_hint

        def insert_prev_op(old_op: Operation, new_op: Operation) -> None:
            """Sets `prev_op` on `self`, and `next_op` on `self.prev_op`."""
            if old_op._prev_op is not None:  # pyright: ignore[reportPrivateUsage]
                # update prev node
                old_op._prev_op._next_op = new_op  # pyright: ignore[reportPrivateUsage]
            # set next and previous on new node
            new_op._prev_op = old_op._prev_op  # pyright: ignore[reportPrivateUsage]
            new_op._next_op = old_op  # pyright: ignore[reportPrivateUsage]
            # update self
            old_op._prev_op = new_op  # pyright: ignore[reportPrivateUsage]

        def insert_op(old_op: Operation, new_op: Operation) -> None:
            """Insert operations at a certain location in a block."""
            new_op.parent = old_op.parent

            prev_op = old_op.prev_op
            insert_prev_op(old_op, new_op)

            if prev_op is None:
                # No `prev_op`, means `next_op` is the first op in the block.
                old_op.parent._first_op = new_op  # pyright: ignore[reportOptionalMemberAccess,reportPrivateUsage]

        def replace_op(
            old_op: Operation, new_op: Operation, new_result: SSAValue
        ) -> None:
            """Replace the matched operation with new operations."""

            # First, insert the new operations before the matched operation
            insert_op(old_op, new_op)

            ## There are no callbacks, so can elide `rewriter.handle_operation_replacement(op_)`

            # Then, replace the results with new ones
            ## We know there is only one result, so can elide the loop
            replace_all_uses_with(old_op.results[0], new_result)
            # Elide "preserv[ing] name hints for ops with multiple results",
            # since done already in `SSAValue.replace_by`

            # Elide "add[ing] name hints for existing ops, only if there is a
            # single new result", since done already in `SSAValue.replace_by`
            erase_op(old_op)

        def construct_int_constant_op(value: int, result_type: type) -> ConstantOp:
            """Efficiently construct and integer constant operation"""
            ## Inline `IntegerAttr(lhs + rhs, result_type)`
            int_attr = IntAttr(value)
            integer_attr = IntegerAttr.__new__(IntegerAttr)
            ## Inline `ParametrizedAttribute.__init__(integer_attr,[int_attr, result_type])`
            object.__setattr__(integer_attr, "parameters", (int_attr, result_type))
            return ConstantOp.create(
                result_types=[result_type],  # pyright: ignore[reportArgumentType]
                properties={"value": integer_attr},
            )

        def has_trait(op: Operation, trait: type[OpTrait]) -> bool:
            """Check if the operation implements a trait with the given parameters."""
            for t in op.traits._traits:  # pyright: ignore
                if isinstance(t, trait):
                    return True
            return False

        def match_and_rewrite(rewrite_op: Operation) -> bool:
            """Match and rewrite an operation."""
            if not isinstance(rewrite_op, AddiOp):
                return False

            lhs_op: OpResult = rewrite_op.operands[0].op  # pyright: ignore
            rhs_op: OpResult = rewrite_op.operands[1].op  # pyright: ignore

            constant_like = ConstantLike
            assert has_trait(lhs_op, constant_like)
            assert has_trait(rhs_op, constant_like)

            lhs: int = lhs_op.value.value.data  # pyright: ignore
            rhs: int = rhs_op.value.value.data  # pyright: ignore
            folded_op = construct_int_constant_op(lhs + rhs, rewrite_op.result.type)  # pyright: ignore

            replace_op(rewrite_op, folded_op, folded_op.results[0])
            return True

        def process_worklist(worklist: list[Operation]) -> bool:
            """Process the worklist until it is empty."""
            rewriter_has_done_action = False

            # Handle empty worklist
            rewrite_op = worklist.pop() if len(worklist) else None
            if rewrite_op is None:
                return rewriter_has_done_action

            # No custom listeners have any effect, as we are operating in the
            # non-recursive mode and no operations are removed in constant folding.
            # As a result of this, we elide `rewriter.extend_from_listener(listener)`.

            while True:
                # Elide exception handling
                rewriter_has_done_action = match_and_rewrite(rewrite_op)

                # If the worklist is empty, we are done
                rewrite_op = worklist.pop() if len(worklist) else None
                if rewrite_op is None:
                    return rewriter_has_done_action

        def populate_worklist(worklist: list[Operation], region: Region) -> None:
            """Populate the worklist with all nested operations."""
            # Elide `for sub_op in region.walk(reverse=True, region_first=True):`
            for sub_block in reversed(region.blocks):
                for sub_op in reversed(sub_block.ops):
                    # Change `self._worklist.push(sub_op)` to native list
                    worklist.append(sub_op)

        region = op.body
        op_was_modified = True
        worklist: list[Operation] = []  # Changed from `Worklist()`
        while op_was_modified:
            populate_worklist(worklist, region)
            op_was_modified = process_worklist(worklist)

    def apply_inlined(self, ctx: Context, op: ModuleOp) -> None:
        """Apply the pass.

        This is a manual inlining of the call stack invoked by:

        ```python
        pattern = ConstantFoldingIntegerAdditionPattern()
        PatternRewriteWalker(pattern).rewrite_module(op)
        ```
        """

        def match_and_rewrite(rewrite_op: Operation) -> bool:
            """Match and rewrite an operation."""
            if not isinstance(rewrite_op, AddiOp):
                return False

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

            result_type = rewrite_op.result.type
            ## Inline `IntegerAttr(lhs + rhs, result_type)`
            int_attr = IntAttr(lhs + rhs)
            integer_attr = IntegerAttr.__new__(IntegerAttr)
            ## Inline `ParametrizedAttribute.__init__(integer_attr,[int_attr, result_type])`
            object.__setattr__(integer_attr, "parameters", (int_attr, result_type))
            folded_op = ConstantOp.create(
                result_types=[result_type],  # pyright: ignore[reportArgumentType]
                properties={"value": integer_attr},
            )

            old_op, new_op, new_result = rewrite_op, folded_op, folded_op.results[0]

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
            for use in old_result.uses.copy():
                ## Inline `use.operation.operands.__setitem__(...)`
                operands = use.operation._operands  # pyright: ignore[reportPrivateUsage]
                ## Inline `operands[use.index].remove_use(Use(use.operation, use.index))`
                operands[use.index].uses.remove(use)
                ## Inline `new_result.add_use(Use(use.operation, use.index))`
                new_result.uses.add(use)
                new_operands = (
                    *operands[: use.index],
                    new_result,
                    *operands[use.index + 1 :],
                )
                use.operation._operands = new_operands  # pyright: ignore[reportPrivateUsage]
            new_result.name_hint = old_result.name_hint

            # Elide "preserv[ing] name hints for ops with multiple results",
            # since done already in `SSAValue.replace_by`

            # Elide "add[ing] name hints for existing ops, only if there is a
            # single new result", since done already in `SSAValue.replace_by`

            ## There are no callbacks, so can elide `rewriter.handle_operation_removal(old_op)`
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
            detached_op = old_op

            detached_op.parent = None
            for idx, operand in enumerate(detached_op._operands):
                ## Inline `operand.remove_use(Use(detached_op, idx))`
                operand.uses.remove(Use(detached_op, idx))
            ## This application has no regions, so no recursive drops

            for result in detached_op.results:
                ## Inline `result.replace_by(ErasedSSAValue(result.type, result))`
                replace_value = ErasedSSAValue(result.type, result)
                ## Newly constructed `ErasedSSAValue`s have no uses, so can
                ## elide copying its uses
                # carry over name if possible
                if replace_value.name_hint is None:
                    replace_value.name_hint = result.name_hint

            return True

        def process_worklist(worklist: list[Operation]) -> bool:
            """Process the worklist until it is empty."""
            rewriter_has_done_action = False

            # Handle empty worklist
            rewrite_op = worklist.pop() if len(worklist) else None
            if rewrite_op is None:
                return rewriter_has_done_action

            # No custom listeners have any effect, as we are operating in the
            # non-recursive mode and no operations are removed in constant folding.
            # As a result of this, we elide `rewriter.extend_from_listener(listener)`.

            while True:
                # Elide exception handling
                rewriter_has_done_action = match_and_rewrite(rewrite_op)

                # If the worklist is empty, we are done
                rewrite_op = worklist.pop() if len(worklist) else None
                if rewrite_op is None:
                    return rewriter_has_done_action

        region = op.body
        op_was_modified = True
        worklist: list[Operation] = []  # Changed from `Worklist()`
        while op_was_modified:
            for sub_block in reversed(region.blocks):
                for sub_op in reversed(sub_block.ops):
                    worklist.append(sub_op)
            op_was_modified = process_worklist(worklist)
