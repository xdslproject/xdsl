from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union, Tuple

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, OpResult, Region, Block, BlockArgument, Attribute
from xdsl.rewriter import Rewriter


@dataclass(eq=False)
class PatternRewriter:
    current_operation: Operation
    has_erased_matched_operation: bool = field(default=False, init=False)
    added_operations: List[Operation] = field(default_factory=list, init=False)
    has_done_action: bool = field(default=False, init=False)

    def _can_modify_op(self, op: Operation) -> bool:
        if op == self.current_operation:
            return True
        if op.parent is None:
            return self.current_operation.get_toplevel_object() is not op
        return self._can_modify_block(op.parent)

    def _can_modify_block(self, block: Block) -> bool:
        if block.parent is None:
            return True  # Toplevel operation of current_operation is always a ModuleOp
        return self._can_modify_region(block.parent)

    def _can_modify_region(self, region: Region) -> bool:
        if region.parent is None:
            return True  # Toplevel operation of current_operation is always a ModuleOp
        return self._can_modify_op(region.parent)

    def erase_matched_op(self):
        self.has_done_action = True
        self.has_erased_matched_operation = True
        Rewriter.erase_op(self.current_operation)

    def erase_op(self, op: Operation):
        self.has_done_action = True
        if op == self.current_operation:
            return self.erase_matched_op()
        if not self._can_modify_op(op):
            raise Exception(
                "PatternRewriter can only erase operations that are the matched operation"
                ", or that are contained in the matched operation.")
        Rewriter.erase_op(op)

    def replace_matched_op(
            self,
            new_ops: Union[Operation, List[Operation]],
            new_results: Optional[List[Optional[OpResult]]] = None,
            safe_erase: bool = True):
        self.has_done_action = True
        if not isinstance(new_ops, list):
            new_ops = [new_ops]
        self.has_erased_matched_operation = True
        Rewriter.replace_op(self.current_operation,
                            new_ops,
                            new_results,
                            safe_erase=safe_erase)
        self.added_operations += new_ops

    def replace_op(self,
                   op: Operation,
                   new_ops: Union[Operation, List[Operation]],
                   new_results: Optional[List[Optional[OpResult]]] = None,
                   safe_erase: bool = True):
        self.has_done_action = True
        if op == self.current_operation:
            return self.replace_matched_op(new_ops, new_results, safe_erase)
        if not self._can_modify_op(op):
            raise Exception(
                "PatternRewriter can only replace operations that are the matched operation"
                ", or that are contained in the matched operation.")
        Rewriter.replace_op(op, new_ops, new_results, safe_erase=safe_erase)

    def modify_block_argument_type(self, arg: BlockArgument,
                                   new_type: Attribute):
        """
        Modify the type of a block argument.
        The block should be contained in the matched operation.
        """
        if not self._can_modify_block(arg.block):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation"
            )
        arg.typ = new_type

    def insert_block_argument(self, block: Block, index: int,
                              typ: Attribute) -> BlockArgument:
        """
        Insert a new block argument.
        The block should be contained in the matched operation.
        """
        if not self._can_modify_block(block):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation"
            )
        return block.insert_arg(typ, index)

    def erase_block_argument(self,
                             arg: BlockArgument,
                             safe_erase: bool = True) -> None:
        """
        Erase a new block argument.
        The block should be contained in the matched operation.
        If safe_erase is true, then raise an exception if the block argument has still uses,
        otherwise, replace it with an ErasedSSAValue.
        """
        if not self._can_modify_block(arg.block):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation"
            )
        arg.block.erase_arg(arg, safe_erase=safe_erase)

    def inline_block_at_pos(self, block: Block, target_block: Block, pos: int):
        self.has_done_action = True
        if not self._can_modify_block(
                target_block) or not self._can_modify_block(block):
            raise Exception(
                "Cannot modify blocks that are not contained in the matched operation."
            )
        Rewriter.inline_block_at_pos(block, target_block, pos)

    def inline_block_before_matched_op(self, block: Block):
        self.has_done_action = True
        if not self._can_modify_block(block):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        self.added_operations += block.ops
        Rewriter.inline_block_before(block, self.current_operation)

    def inline_block_before(self, block: Block, op: Operation):
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
                " or before an operation child")
        Rewriter.inline_block_before(block, op)

    def inline_block_after(self, block: Block, op: Operation):
        self.has_done_action = True
        if op is self.current_operation:
            return self.inline_block_before_matched_op(block)
        if not self._can_modify_block(block) or not self._can_modify_block(
                op.parent):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        Rewriter.inline_block_after(block, op)


class RewritePattern(ABC):
    """
    A side-effect free rewrite pattern matching on a DAG.
    """

    @abstractmethod
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        """
        Match an operation, and optionally perform a rewrite using the rewriter.
        """
        ...


@dataclass(eq=False, repr=False)
class AnonymousRewritePattern(RewritePattern):
    """
    A rewrite pattern encoded by an anonymous function.
    """
    func: Callable[[Operation, PatternRewriter], None]

    def match_and_rewrite(self, op: Operation,
                          rewriter: PatternRewriter) -> None:
        self.func(op, rewriter)


def op_type_rewrite_pattern(func):
    """
    This function is intended to be used as a decorator on a RewritePatter method.
    It uses type hints to match on a specific operation type before calling the decorated function.
    """
    # Get the operation argument and check that it is a subclass of Operation
    params = [param for param in inspect.signature(func).parameters.values()]
    if len(params) < 2:
        raise Exception(
            "op_type_rewrite_pattern expects the decorated function to "
            "have two non-self arguments.")
    is_method = params[0].name == "self"
    if is_method:
        if len(params) != 3:
            raise Exception(
                "op_type_rewrite_pattern expects the decorated method to "
                "have two non-self arguments.")
    else:
        if len(params) != 2:
            raise Exception(
                "op_type_rewrite_pattern expects the decorated function to "
                "have two arguments.")
    expected_type = params[-2].annotation
    if not issubclass(expected_type, Operation):
        raise Exception(
            "op_type_rewrite_pattern expects the first non-self argument"
            "type hint to be an Operation subclass")

    if not is_method:

        def op_type_rewrite_pattern_static_wrapper(
                op: Operation, rewriter: PatternRewriter) -> None:
            if not isinstance(op, expected_type):
                return None
            func(op, rewriter)

        return op_type_rewrite_pattern_static_wrapper

    def op_type_rewrite_pattern_method_wrapper(
            self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, expected_type):
            return None
        func(self, op, rewriter)

    return op_type_rewrite_pattern_method_wrapper


@dataclass(eq=False, repr=False)
class GreedyRewritePatternApplier(RewritePattern):
    """Apply a list of patterns in order until one pattern matches, and then use this rewrite."""

    rewrite_patterns: List[RewritePattern]
    """The list of rewrites to apply in order."""

    def match_and_rewrite(self, op: Operation,
                          rewriter: PatternRewriter) -> None:
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
    """Choose if the walker should first walk the operation regions first, or the operation itself."""

    apply_recursively: bool = field(default=True)
    """Apply recursively rewrites on new operations."""

    walk_reverse: bool = field(default=False)
    """Walk the regions and blocks in reverse order. That way, all uses are replaced before the definitions."""

    def rewrite_module(self, op: ModuleOp):
        """Rewrite an entire module operation."""
        self._rewrite_op(op)

    def _rewrite_op(self, op: Operation) -> int:
        """Rewrite an operation, along with its regions."""
        # First, we rewrite the regions if needed
        if self.walk_regions_first:
            self._rewrite_op_regions(op)

        # We then match for a pattern in the current operation
        rewriter = PatternRewriter(op)
        self.pattern.match_and_rewrite(op, rewriter)

        if rewriter.has_done_action:
            # If we produce new operations, we rewrite them recursively if requested
            if self.apply_recursively:
                return len(rewriter.added_operations) - int(
                    rewriter.has_erased_matched_operation
                ) if self.walk_reverse else 0
            # Else, we rewrite only their regions if they are supposed to be rewritten after
            else:
                if not self.walk_regions_first:
                    for new_op in rewriter.added_operations:
                        self._rewrite_op_regions(new_op)
                    if not rewriter.has_erased_matched_operation:
                        self._rewrite_op_regions(op)
                return -1 if self.walk_reverse else len(
                    rewriter.added_operations) + int(
                        not rewriter.has_erased_matched_operation)

        # Otherwise, we only rewrite the regions of the operation if needed
        if not self.walk_regions_first:
            self._rewrite_op_regions(op)
        return -1 if self.walk_reverse else 1

    def _rewrite_op_regions(self, op: Operation):
        """Rewrite the regions of an operation, and update the operation with the new regions."""
        if not self.walk_reverse:
            for region in op.regions:
                for block in region.blocks:
                    idx = 0
                    while idx < len(block.ops):
                        idx += self._rewrite_op(block.ops[idx])
        else:
            for region in op.regions:
                for block in reversed(region.blocks):
                    idx = len(block.ops) - 1
                    while idx >= 0:
                        idx += self._rewrite_op(block.ops[idx])
