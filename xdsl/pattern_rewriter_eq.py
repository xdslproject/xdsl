from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from xdsl.builder import Builder, InsertOpInvT
from xdsl.dialects import equivalence
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    OpResult,
    Region,
    SSAValue,
    Use,
)
from xdsl.pattern_rewriter import PatternRewriterListener
from xdsl.rewriter import BlockInsertPoint, InsertPoint, Rewriter
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet


@dataclass(eq=False, init=False)
class EquivalencePatternRewriter(Builder, PatternRewriterListener):
    """
    A rewriter used during pattern matching.
    Once an operation is matched, this rewriter is used to apply
    modification to the operation and its children.
    """

    # operations that already have an eclass
    known_ops: KnownOps = field(default_factory=KnownOps)
    """Used for hashconsing operations. When new operations are created, if they are identical to an existing operation,
    the existing operation is reused instead of creating a new one."""

    eclass_union_find: DisjointSet[equivalence.AnyClassOp] = field(
        default_factory=lambda: DisjointSet[equivalence.AnyClassOp]()
    )
    """Union-find structure tracking which e-classes are equivalent and should be merged."""

    current_operation: Operation
    """The matched operation."""

    has_done_action: bool = field(default=False, init=False)
    """Has the rewriter done any action during the current match."""

    def get_or_create_class(self, op: InsertOpInvT) -> equivalence.AnyClassOp:
        """
        Get the equivalence class for a value, creating one if it doesn't exist.
        """
        # op is already a ClassOp
        if isinstance(op, equivalence.AnyClassOp):
            return self.eclass_union_find.find(op)

        # op is not a ClassOp, create a ClassOp wrapper for it and check if there exist equivalent ClassOps
        # – if so, union the new ClassOp with the existing one, otherwise, add the new ClassOp to the union-find structure
        eclass_op = equivalence.ClassOp(
            op.results[0]
        )  # creeate a ClassOp wrapper for the operation result
        if op not in self.known_ops:
            self.known_ops[op] = (
                op  # update the known_ops to indicate that we have seen this operation
            )
            self.eclass_union_find.add(
                eclass_op
            )  # add the new ClassOp to the union-find structure
            return eclass_op
        else:  # op has been seen before, so there must exist a ClassOp wrapper already, so we union
            eclass_op = equivalence.ClassOp(op.results[0])
            self.eclass_union_find.find(eclass_op)
            return eclass_op

    def __init__(self, current_operation: Operation):
        PatternRewriterListener.__init__(self)
        self.current_operation = current_operation
        Builder.__init__(self, InsertPoint.before(current_operation))

    # change this to insert eq class & deduplicate operation
    # check if an eclass for the operation already exists, if so resues it, if not create one
    def insert_op(
        self,
        op: InsertOpInvT,
        insertion_point: InsertPoint | None = None,
    ) -> InsertOpInvT:
        """Insert operations at a certain location in a block."""
        eclass_op = self.get_or_create_class(op)
        self.has_done_action = True
        return super().insert_op(eclass_op, insertion_point)

    # do i need to edit this?
    def erase_op(self, op: Operation, safe_erase: bool = True):
        """
        Erase an operation.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
        self.handle_operation_removal(op)
        Rewriter.erase_op(op, safe_erase=safe_erase)

    def replace_all_uses_with(
        self, from_: SSAValue, to: SSAValue | None, safe_erase: bool = True
    ):
        """Replace all uses of an SSA value with another SSA value."""
        modified_ops = [use.operation for use in from_.uses]
        if to is None:
            from_.erase(safe_erase=safe_erase)
        else:
            from_.replace_by(to)
        for op in modified_ops:
            self.handle_operation_modification(op)

    def replace_uses_with_if(
        self,
        from_: SSAValue,
        to: SSAValue,
        predicate: Callable[[Use], bool],
    ):
        """Find uses of from and replace them with to if the predicate returns true."""
        uses_to_replace = [use for use in from_.uses if predicate(use)]
        modified_ops = [use.operation for use in uses_to_replace]

        for use in uses_to_replace:
            use.operation.operands[use.index] = to

        for op in modified_ops:
            self.handle_operation_modification(op)

    def replace_matched_op(
        self,
        new_ops: Operation | Sequence[Operation],
        new_results: Sequence[SSAValue | None] | None = None,
        safe_erase: bool = True,
    ):
        """
        Replace the matched operation with new operations.
        Also, optionally specify SSA values to replace the operation results.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.replace_op(
            self.current_operation, new_ops, new_results, safe_erase=safe_erase
        )

    # do not remove the operation.
    # If both the current operation and new operation do not have an eclass yet, create on,
    # otherwise, add the new operation to the existing eqclass/ union the operation with its eclass
    # replace current operation with the eclass.
    def replace_op(
        self,
        op: Operation,
        new_ops: Operation | Sequence[Operation],
        new_results: Sequence[SSAValue | None] | None = None,
        safe_erase: bool = True,
    ):
        """
        Replace an operation with new operations.
        Also, optionally specify SSA values to replace the operation results.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True

        if isinstance(new_ops, Operation):
            new_ops = (new_ops,)

        # First, insert the new operations before the matched operation
        self.insert_op(new_ops, InsertPoint.before(op))

        if new_results is None:
            new_results = new_ops[-1].results if new_ops else []

        if len(op.results) != len(new_results):
            raise ValueError(
                f"Expected {len(op.results)} new results, but got {len(new_results)}"
            )

        # Then, replace the results with new ones
        self.handle_operation_replacement(op, new_results)
        for old_result, new_result in zip(op.results, new_results):
            self.replace_all_uses_with(old_result, new_result, safe_erase=safe_erase)

            # Preserve name hints for ops with multiple results
            if new_result is not None and not new_result.name_hint:
                new_result.name_hint = old_result.name_hint

        # Add name hints for existing ops, only if there is a single new result
        if (
            len(new_results) == 1
            and (only_result := new_results[0]) is not None
            and (name_hint := only_result.name_hint) is not None
        ):
            for new_op in new_ops:
                for res in new_op.results:
                    if not res.name_hint:
                        res.name_hint = name_hint

        # Then, erase the original operation
        self.erase_op(op, safe_erase=safe_erase)

    def replace_value_with_new_type(
        self, val: SSAValue, new_type: Attribute
    ) -> SSAValue:
        """
        Replace a value with a value of a new type, and return the new value.
        This will insert the new value in the operation or block, and remove the existing
        value.
        """
        self.has_done_action = True
        if isinstance(val, OpResult):
            self.handle_operation_modification(val.op)
        if isinstance(val, BlockArgument):
            if (op := val.block.parent_op()) is not None:
                self.handle_operation_modification(op)
        return Rewriter.replace_value_with_new_type(val, new_type)

    def insert_block_argument(
        self, block: Block, index: int, arg_type: Attribute
    ) -> BlockArgument:
        """Insert a new block argument."""
        self.has_done_action = True
        return block.insert_arg(arg_type, index)

    def erase_block_argument(self, arg: BlockArgument, safe_erase: bool = True) -> None:
        """
        Erase a new block argument.
        If safe_erase is true, then raise an exception if the block argument has still
        uses, otherwise, replace it with an ErasedSSAValue.
        """
        self.has_done_action = True
        self.replace_all_uses_with(arg, None, safe_erase=safe_erase)
        arg.block.erase_arg(arg, safe_erase)

    def inline_block(
        self,
        block: Block,
        insertion_point: InsertPoint,
        arg_values: Sequence[SSAValue] = (),
    ):
        """
        Move the block operations to the specified insertion point.
        """
        self.has_done_action = True
        Rewriter.inline_block(block, insertion_point, arg_values=arg_values)

    def move_region_contents_to_new_regions(self, region: Region) -> Region:
        """Move the region blocks to a new region."""
        self.has_done_action = True
        return Rewriter.move_region_contents_to_new_regions(region)

    def inline_region(self, region: Region, insertion_point: BlockInsertPoint) -> None:
        """Move the region blocks to the specified insertion point."""
        self.has_done_action = True
        Rewriter.inline_region(region, insertion_point)

    def notify_op_modified(self, op: Operation) -> None:
        """
        Notify the rewriter that an operation was modified in the pattern.
        This will correctly update the rewriter state.
        """
        self.has_done_action = True
        self.handle_operation_modification(op)
