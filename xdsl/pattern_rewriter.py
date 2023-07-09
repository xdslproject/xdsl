from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import UnionType
from typing import Callable, Iterable, Sequence, TypeVar, Union, get_args, get_origin

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Block, BlockArgument, Operation, Region, SSAValue
from xdsl.rewriter import Rewriter


@dataclass(eq=False)
class PatternRewriter:
    """
    A rewriter used during pattern matching.
    Once an operation is matched, this rewriter is used to apply
    modification to the operation and its children.
    """

    current_operation: Operation
    """The matched operation."""

    has_erased_matched_operation: bool = field(default=False, init=False)
    """Was the matched operation erased."""

    added_operations_before: list[Operation] = field(default_factory=list, init=False)
    """The operations added directly before the matched operation."""

    added_operations_after: list[Operation] = field(default_factory=list, init=False)
    """The operations added directly after the matched operation."""

    has_done_action: bool = field(default=False, init=False)
    """Has the rewriter done any action during the current match."""

    def _can_modify_op(self, op: Operation) -> bool:
        """Check if the operation and its children can be modified by this rewriter."""
        if op == self.current_operation:
            return True
        if op.parent is None:
            return self.current_operation.get_toplevel_object() is not op
        return self._can_modify_op_in_block(op.parent)

    def _can_modify_block(self, block: Block) -> bool:
        """Check if the block can be modified by this rewriter."""
        if block is self.current_operation.parent:
            return True
        return self._can_modify_op_in_block(block)

    def _can_modify_op_in_block(self, block: Block) -> bool:
        """Check if the block and its children can be modified by this rewriter."""
        if block.parent is None:
            return True  # Toplevel operation of current_operation is always a ModuleOp
        return self._can_modify_region(block.parent)

    def _can_modify_region(self, region: Region) -> bool:
        """Check if the region and its children can be modified by this rewriter."""
        if region.parent is None:
            return True  # Toplevel operation of current_operation is always a ModuleOp
        return self._can_modify_op(region.parent)

    def insert_op_before_matched_op(self, op: (Operation | Sequence[Operation])):
        """Insert operations before the matched operation."""
        if self.current_operation.parent is None:
            raise Exception("Cannot insert an operation before a toplevel operation.")
        self.has_done_action = True
        block = self.current_operation.parent
        op = [op] if isinstance(op, Operation) else op
        if len(op) == 0:
            return
        block.insert_ops_before(op, self.current_operation)
        self.added_operations_before.extend(op)

    def insert_op_after_matched_op(self, op: (Operation | Sequence[Operation])):
        """Insert operations after the matched operation."""
        if self.current_operation.parent is None:
            raise Exception("Cannot insert an operation after a toplevel operation.")
        self.has_done_action = True
        block = self.current_operation.parent
        op = [op] if isinstance(op, Operation) else op
        if len(op) == 0:
            return
        block.insert_ops_after(op, self.current_operation)
        self.added_operations_after.extend(op)

    def insert_op_at_end(self, op: Operation | Sequence[Operation], block: Block):
        """Insert operations in a block contained in the matched operation."""
        if not self._can_modify_block(block):
            raise Exception("Cannot insert operations in block.")
        self.has_done_action = True
        op = [op] if isinstance(op, Operation) else op
        if len(op) == 0:
            return
        block.add_ops(op)

    def insert_op_at_start(self, op: Operation | Sequence[Operation], block: Block):
        """Insert operations in a block contained in the matched operation."""
        if not self._can_modify_block(block):
            raise Exception("Cannot insert operations in block.")
        first_op = block.first_op
        if first_op is None:
            self.insert_op_at_end(op, block)
        else:
            self.insert_op_before(op, first_op)

    def insert_op_before(
        self, op: Operation | Sequence[Operation], target_op: Operation
    ):
        """Insert operations before an operation contained in the matched operation."""
        if target_op.parent is None:
            raise Exception("Cannot insert operations before toplevel operation.")
        target_block = target_op.parent
        if not self._can_modify_block(target_block):
            raise Exception("Cannot insert operations in this block.")
        self.has_done_action = True
        op = [op] if isinstance(op, Operation) else op
        if len(op) == 0:
            return
        target_block.insert_ops_before(op, target_op)

    def insert_op_after(
        self, op: Operation | Sequence[Operation], target_op: Operation
    ):
        """Insert operations after an operation contained in the matched operation."""
        if target_op.parent is None:
            raise Exception("Cannot insert operations after toplevel operation.")
        target_block = target_op.parent
        if not self._can_modify_block(target_block):
            raise Exception("Cannot insert operations in this block.")
        self.has_done_action = True
        ops = [op] if isinstance(op, Operation) else op
        if len(ops) == 0:
            return
        target_block.insert_ops_after(ops, target_op)

    def erase_matched_op(self, safe_erase: bool = True):
        """
        Erase the operation that was matched to.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
        self.has_erased_matched_operation = True
        Rewriter.erase_op(self.current_operation, safe_erase=safe_erase)

    def erase_op(self, op: Operation, safe_erase: bool = True):
        """
        Erase an operation contained in the matched operation children.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
        if op == self.current_operation:
            return self.erase_matched_op(safe_erase)
        if not self._can_modify_op(op):
            raise Exception(
                "PatternRewriter can only erase operations that are the matched operation"
                ", or that are contained in the matched operation."
            )
        Rewriter.erase_op(op, safe_erase=safe_erase)

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
        self.has_done_action = True
        if isinstance(new_ops, Operation):
            new_ops = [new_ops]
        self.has_erased_matched_operation = True
        Rewriter.replace_op(
            self.current_operation, new_ops, new_results, safe_erase=safe_erase
        )
        self.added_operations_before.extend(new_ops)

    def replace_op(
        self,
        op: Operation,
        new_ops: Operation | Sequence[Operation],
        new_results: Sequence[SSAValue | None] | None = None,
        safe_erase: bool = True,
    ):
        """
        Replace an operation with new operations.
        The operation should be a child of the matched operation.
        Also, optionally specify SSA values to replace the operation results.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
        if op == self.current_operation:
            return self.replace_matched_op(new_ops, new_results, safe_erase)
        if not self._can_modify_op(op):
            raise Exception(
                "PatternRewriter can only replace operations that are the matched "
                "operation, or that are contained in the matched operation."
            )
        Rewriter.replace_op(op, new_ops, new_results, safe_erase=safe_erase)

    def modify_block_argument_type(self, arg: BlockArgument, new_type: Attribute):
        """
        Modify the type of a block argument.
        The block should be contained in the matched operation.
        """
        if not self._can_modify_block(arg.block):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation"
            )
        self.has_done_action = True
        arg.type = new_type

    def insert_block_argument(
        self, block: Block, index: int, typ: Attribute
    ) -> BlockArgument:
        """
        Insert a new block argument.
        The block should be contained in the matched operation.
        """
        if not self._can_modify_block(block):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation"
            )
        self.has_done_action = True
        return block.insert_arg(typ, index)

    def erase_block_argument(self, arg: BlockArgument, safe_erase: bool = True) -> None:
        """
        Erase a new block argument.
        The block should be contained in the matched operation.
        If safe_erase is true, then raise an exception if the block argument has still
        uses, otherwise, replace it with an ErasedSSAValue.
        """
        if not self._can_modify_block(arg.block):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation"
            )
        self.has_done_action = True
        arg.block.erase_arg(arg, safe_erase=safe_erase)

    def inline_block_at_end(self, block: Block, target_block: Block):
        """
        Move the block operations to the end of another block.
        This block should not be a parent of the block to move to.
        """
        self.has_done_action = True
        if not self._can_modify_block(target_block) or not self._can_modify_block(
            block
        ):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation."
            )
        Rewriter.inline_block_at_end(block, target_block)

    def inline_block_at_start(self, block: Block, target_block: Block):
        """
        Move the block operations to the start of another block.
        This block should not be a parent of the block to move to.
        """
        self.has_done_action = True
        if not self._can_modify_block(target_block) or not self._can_modify_block(
            block
        ):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation."
            )
        Rewriter.inline_block_at_start(block, target_block)

    def inline_block_before_matched_op(self, block: Block):
        """
        Move the block operations before the matched operation.
        The block should not be a parent of the operation, and should be a child of the
        matched operation.
        """
        self.has_done_action = True
        if not self._can_modify_block(block):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        self.added_operations_before.extend(block.ops)
        Rewriter.inline_block_before(block, self.current_operation)

    def inline_block_before(self, block: Block, op: Operation):
        """
        Move the block operations before the given operation.
        The block should not be a parent of the operation, and should be a child of the
        matched operation.
        The operation should also be a child of the matched operation.
        """
        self.has_done_action = True
        if op is self.current_operation:
            return self.inline_block_before_matched_op(block)
        if not self._can_modify_block(block):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        if not self._can_modify_op(op):
            raise Exception(
                "Cannot move block elsewhere than before the matched operation,"
                " or before an operation child"
            )
        Rewriter.inline_block_before(block, op)

    def inline_block_after_matched_op(self, block: Block):
        """
        Move the block operations after the matched operation.
        The block should not be a parent of the operation, and should be a child of the
        matched operation.
        """
        self.has_done_action = True
        if not self._can_modify_block(block):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        self.added_operations_after.extend(block.ops)
        Rewriter.inline_block_after(block, self.current_operation)

    def inline_block_after(self, block: Block, op: Operation):
        """
        Move the block operations after the given operation.
        The block should not be a parent of the operation, and should be a child of the
        matched operation.
        The operation should also be a child of the matched operation.
        """
        self.has_done_action = True
        if op is self.current_operation:
            return self.inline_block_after_matched_op(block)
        if not self._can_modify_block(block) or (
            op.parent is not None and not self._can_modify_block(op.parent)
        ):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        Rewriter.inline_block_after(block, op)

    def move_region_contents_to_new_regions(self, region: Region) -> Region:
        """
        Move the region blocks to a new region.
        The region should be a child of the matched operation.
        """
        self.has_done_action = True
        if not self._can_modify_region(region):
            raise Exception(
                "Cannot move regions that are not children of the matched operation"
            )
        return Rewriter.move_region_contents_to_new_regions(region)

    def iter_affected_ops(self) -> Iterable[Operation]:
        """
        Iterate newly added operations, in the order that they are in the module.
        """
        yield from self.added_operations_before
        if not self.has_erased_matched_operation:
            yield self.current_operation
        yield from self.added_operations_after

    def iter_affected_ops_reversed(self) -> Iterable[Operation]:
        """
        Iterate newly added operations, in reverse order from that in the module.
        """
        yield from reversed(self.added_operations_after)
        if not self.has_erased_matched_operation:
            yield self.current_operation
        yield from reversed(self.added_operations_before)


class RewritePattern(ABC):
    """
    A side-effect free rewrite pattern matching on a DAG.
    """

    # The / in the function signature makes the previous arguments positional, see
    # https://peps.python.org/pep-0570/
    # This is used by the op_type_rewrite_pattern
    @abstractmethod
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        """
        Match an operation, and optionally perform a rewrite using the rewriter.
        """
        ...


_RewritePatternT = TypeVar("_RewritePatternT", bound=RewritePattern)
_OperationT = TypeVar("_OperationT", bound=Operation)


def op_type_rewrite_pattern(
    func: Callable[[_RewritePatternT, _OperationT, PatternRewriter], None]
) -> Callable[[_RewritePatternT, Operation, PatternRewriter], None]:
    """
    This function is intended to be used as a decorator on a RewritePatter
    method. It uses type hints to match on a specific operation type before
    calling the decorated function.
    """
    # Get the operation argument and check that it is a subclass of Operation
    params = [param for param in inspect.signature(func).parameters.values()]
    if len(params) != 3:
        raise Exception(
            "op_type_rewrite_pattern expects the decorated function to "
            "have two non-self arguments."
        )
    is_method = params[0].name == "self"
    if is_method:
        if len(params) != 3:
            raise Exception(
                "op_type_rewrite_pattern expects the decorated method to "
                "have two non-self arguments."
            )
    else:
        if len(params) != 2:
            raise Exception(
                "op_type_rewrite_pattern expects the decorated function to "
                "have two arguments."
            )
    expected_type: type[_OperationT] = params[-2].annotation

    expected_types = (expected_type,)
    if get_origin(expected_type) in [Union, UnionType]:
        expected_types = get_args(expected_type)

    if not all(issubclass(t, Operation) for t in expected_types):
        raise Exception(
            "op_type_rewrite_pattern expects the first non-self argument "
            "type hint to be an `Operation` subclass or a union of `Operation` "
            "subclasses."
        )

    def impl(self: _RewritePatternT, op: Operation, rewriter: PatternRewriter) -> None:
        if isinstance(op, expected_type):
            func(self, op, rewriter)

    return impl


@dataclass(eq=False, repr=False)
class GreedyRewritePatternApplier(RewritePattern):
    """
    Apply a list of patterns in order until one pattern matches,
    and then use this rewrite.
    """

    rewrite_patterns: list[RewritePattern]
    """The list of rewrites to apply in order."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        for pattern in self.rewrite_patterns:
            pattern.match_and_rewrite(op, rewriter)
            if rewriter.has_done_action:
                return
        return


@dataclass(eq=False, repr=False)
class PatternRewriteWalker:
    """
    Walks the IR in the block and instruction order, and rewrite it in place.
    Previous references to the walked operations are invalid after the walk.
    Can walk either first the regions, or first the owner operation.
    The walker will also walk recursively on the created operations.
    """

    pattern: RewritePattern
    """Pattern to apply during the walk."""

    walk_regions_first: bool = field(default=False)
    """
    Choose if the walker should first walk the operation regions first,
    or the operation itself.
    """

    apply_recursively: bool = field(default=True)
    """Apply recursively rewrites on new operations."""

    walk_reverse: bool = field(default=False)
    """
    Walk the regions and blocks in reverse order.
    That way, all uses are replaced before the definitions.
    """

    def rewrite_module(self, op: ModuleOp):
        """Rewrite an entire module operation."""
        self._rewrite_op(op)

    def _rewrite_op(self, op: Operation) -> Operation | None:
        """
        Rewrite an operation, along with its regions.
        Returns the next operation to iterate over.
        """
        # First, we rewrite the regions if needed
        if self.walk_regions_first:
            self._rewrite_op_regions(op)

        prev_op = op.prev_op
        next_op = op.next_op

        # We then match for a pattern in the current operation
        rewriter = PatternRewriter(op)
        self.pattern.match_and_rewrite(op, rewriter)

        if rewriter.has_done_action:
            # If we produce new operations, we rewrite them recursively if requested
            if self.apply_recursively:
                if self.walk_reverse:
                    for op in rewriter.iter_affected_ops_reversed():
                        # return last affected op
                        return op
                    else:
                        return prev_op
                else:
                    for op in rewriter.iter_affected_ops():
                        # return first affected op
                        return op
                    else:
                        return next_op

            # Else, we rewrite only their regions if they are supposed to be
            # rewritten after
            else:
                if not self.walk_regions_first:
                    for new_op in rewriter.added_operations_before:
                        self._rewrite_op_regions(new_op)
                    if not rewriter.has_erased_matched_operation:
                        self._rewrite_op_regions(op)
                    for new_op in rewriter.added_operations_after:
                        self._rewrite_op_regions(new_op)
                return prev_op if self.walk_reverse else next_op

        # Otherwise, we only rewrite the regions of the operation if needed
        if not self.walk_regions_first:
            self._rewrite_op_regions(op)
        return prev_op if self.walk_reverse else next_op

    def _rewrite_op_regions(self, op: Operation):
        """
        Rewrite the regions of an operation, and update the operation with the
        new regions.
        """
        for region in op.regions:
            blocks = reversed(region.blocks) if self.walk_reverse else region.blocks
            for block in blocks:
                iter_op = block.last_op if self.walk_reverse else block.first_op
                while iter_op is not None:
                    iter_op = self._rewrite_op(iter_op)
