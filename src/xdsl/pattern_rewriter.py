from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, OpResult
from xdsl.rewriter import Rewriter


@dataclass(eq=False)
class PatternRewriter:
    current_operation: Operation
    has_done_action: bool = field(default=False, init=False)
    added_operations: List[Operation] = field(default_factory=list)

    def _check_can_act(self) -> None:
        if self.has_done_action:
            raise ValueError(
                "Cannot replace or erase multiple time in the same match")

    def erase_op(self):
        self._check_can_act()
        Rewriter.erase_op(self.current_operation)
        self.has_done_action = True

    def replace_op(self,
                   new_ops: Union[Operation, List[Operation]],
                   new_results: Optional[List[OpResult]] = None):
        if not isinstance(new_ops, list):
            new_ops = [new_ops]
        self._check_can_act()
        Rewriter.replace_op(self.current_operation, new_ops, new_results)
        self.added_operations += new_ops
        self.has_done_action = True


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
    """Apply a list of patterns in order until one pattern match, and then use this rewrite."""

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
                return 0
            # Else, we rewrite only their regions if they are supposed to be rewritten after
            else:
                for new_op in rewriter.added_operations:
                    if not self.walk_regions_first:
                        self._rewrite_op_regions(new_op)
                return len(rewriter.added_operations)

        # Otherwise, we only rewrite the regions of the operation if needed
        if not self.walk_regions_first:
            self._rewrite_op_regions(op)
        return 1

    def _rewrite_op_regions(self, op: Operation):
        """Rewrite the regions of an operation, and update the operation with the new regions."""
        for region in op.regions:
            for block in region.blocks:
                idx = 0
                while idx < len(block.ops):
                    idx += self._rewrite_op(block.ops[idx])
