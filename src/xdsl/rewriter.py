from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block, BlockArgument


@dataclass(eq=False, repr=False)
class RewriteAction:
    """
    Action that a single rewrite may execute.
    A rewrite always delete the matched operation, and replace it with new operations.
    The matched operation results are replaced with new ones.
    """

    new_ops: List[Operation]
    """New operations that replace the one matched."""

    new_results: List[Optional[SSAValue]]
    """SSA values that replace the matched operation results. None values are deleted SSA Values."""
    @staticmethod
    def from_op_list(new_ops: List[Operation]) -> RewriteAction:
        """
        Case where the old results will be replaced by the results of the last operation to be added.
        Can also be used with no operations (to represent deletion).
        """
        if len(new_ops) == 0:
            return RewriteAction(new_ops, [])
        else:
            return RewriteAction(new_ops, new_ops[-1].results)


class RewritePattern(ABC):
    """
    A side-effect free rewrite pattern matching on a DAG.
    """
    @abstractmethod
    def match_and_rewrite(
            self, op: Operation,
            new_operands: List[Optional[SSAValue]]) -> Optional[RewriteAction]:
        """
        Match an operation, and optionally returns a rewrite to be performed.
        `op` is the operation to match, and `new_operands` are the potential new values of the operands.
        `None` values in new_operands are deleted SSA values.
        This function returns `None` if the pattern did not match, and a rewrite action otherwise.
        """
        ...


@dataclass(eq=False, repr=False)
class AnonymousRewritePattern(RewritePattern):
    """
    A rewrite pattern encoded by an anonymous function.
    """
    func: Callable[[Operation, List[Optional[SSAValue]]],
                   Optional[RewriteAction]]

    def match_and_rewrite(
            self, op: Operation,
            new_operands: List[Optional[SSAValue]]) -> Optional[RewriteAction]:
        return self.func(op, new_operands)


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
                op: Operation,
                operands: List[Optional[SSAValue]]) -> Optional[RewriteAction]:
            if not isinstance(op, expected_type):
                return None
            return func(op, operands)

        return op_type_rewrite_pattern_static_wrapper

    def op_type_rewrite_pattern_method_wrapper(
            self, op: Operation,
            operands: List[Optional[SSAValue]]) -> Optional[RewriteAction]:
        if not isinstance(op, expected_type):
            return None
        return func(self, op, operands)

    return op_type_rewrite_pattern_method_wrapper


@dataclass(eq=False, repr=False)
class GreedyRewritePatternApplier(RewritePattern):
    """Apply a list of patterns in order until one pattern match, and then use this rewrite."""

    rewrite_patterns: List[RewritePattern]
    """The list of rewrites to apply in order."""
    def match_and_rewrite(
            self, op: Operation,
            new_operands: List[SSAValue]) -> Optional[RewriteAction]:
        for pattern in self.rewrite_patterns:
            res = pattern.match_and_rewrite(op, new_operands)
            if res is not None:
                return res

        return None


@dataclass(repr=False, eq=False)
class OperandUpdater:
    """
    Provides functionality to bookkeep changed results and to access and update them.
    """

    result_mapping: Dict[SSAValue,
                         Optional[SSAValue]] = field(default_factory=dict)
    """Map old ssa values to new values. Deleted values are mapped to None."""
    def bookkeep_results(self, old_op: Operation,
                         action: RewriteAction) -> None:
        """Bookkeep the changes made by a rewrite action matching on `old_op`."""
        if len(old_op.results) == 0:
            return

        assert len(old_op.results) == len(action.new_results)

        for (old_res, new_res) in zip(old_op.results, action.new_results):
            self.result_mapping[old_res] = new_res

    def bookkeep_blockargs(self, old_block: Block, new_block: Block):
        """ Map old block arguments to the new arguments """
        for (old_arg, new_arg) in zip(old_block.args, new_block.args):
            self.result_mapping[old_arg] = new_arg

    def get_new_value(self, value: SSAValue) -> Optional[SSAValue]:
        """Get the updated value, if it exists, or returns the same one."""
        return self.result_mapping.get(value, value)

    def get_new_operands(self, op: Operation) -> [Optional[SSAValue]]:
        """Get the new operation updated operands"""
        return [self.get_new_value(operand) for operand in op.operands]

    def update_operands(self, op: Operation) -> None:
        """Update an operation operands with the new operands."""
        new_operands = self.get_new_operands(op)
        if None in new_operands:
            raise Exception("Use of deleted SSA Value")
        op.operands = new_operands


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

    _updater: OperandUpdater = field(init=False,
                                     default_factory=OperandUpdater)
    """Takes care of bookkeeping the changes made during the walk."""
    def rewrite_module(self, op: ModuleOp):
        """Rewrite an entire module operation."""
        new_ops = self.rewrite_op(op)
        if len(new_ops) == 1:
            res_op = new_ops[0]
            if isinstance(res_op, ModuleOp):
                op = res_op
                return
        raise Exception(
            "Rewrite pattern did not rewrite a module into another module.")

    def rewrite_op(self, op: Operation) -> List[Operation]:
        """Rewrite an operation, along with its regions."""
        # First, we walk the regions if needed
        if self.walk_regions_first:
            self.rewrite_op_regions(op)

        # We then match for a pattern in the current operation
        action = self.pattern.match_and_rewrite(
            op, self._updater.get_new_operands(op))

        # If we produce new operations, we rewrite them recursively if requested
        if action is not None:
            self._updater.bookkeep_results(op, action)
            if self.apply_recursively:
                new_ops = []
                for new_op in action.new_ops:
                    new_ops.extend(self.rewrite_op(new_op))
                return new_ops
            else:
                for new_op in action.new_ops:
                    self._updater.update_operands(new_op)
                return action.new_ops

        # Otherwise, we update their operands, and walk recursively their regions if needed
        self._updater.update_operands(op)
        if not self.walk_regions_first:
            self.rewrite_op_regions(op)
        return [op]

    def add_block_args(self, new: Block, old: Block):
        """Duplicate the old BlockArguments, adds them to the new Block, and bookkeeps them."""
        new.args = [BlockArgument(arg.typ, new, arg.index) for arg in old.args]
        self._updater.bookkeep_blockargs(old, new)

    def rewrite_op_regions(self, op: Operation):
        """Rewrite the regions of an operation, and update the operation with the new regions."""
        new_regions = []
        for region in op.regions:
            new_region = Region()
            for block in region.blocks:
                new_block = Block()
                self.add_block_args(new_block, block)
                for sub_op in block.ops:
                    new_block.add_ops(self.rewrite_op(sub_op))
                new_region.add_block(new_block)
            new_regions.append(new_region)
        op.regions = []
        for region in new_regions:
            op.add_region(region)
