from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import wraps
from types import UnionType
from typing import (
    TypeVar,
    Union,
    final,
    get_args,
    get_origin,
)

from xdsl.builder import BuilderListener
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.rewriter import Rewriter
from xdsl.utils.hints import isa


@dataclass(eq=False)
class PatternRewriterListener(BuilderListener):
    """A listener for pattern rewriter events."""

    operation_removal_handler: list[Callable[[Operation], None]] = field(
        default_factory=list, kw_only=True
    )
    """Callbacks that are called when an operation is removed."""

    operation_modification_handler: list[Callable[[Operation], None]] = field(
        default_factory=list,
        kw_only=True,
    )
    """Callbacks that are called when an operation is modified."""

    operation_replacement_handler: list[
        Callable[[Operation, Sequence[SSAValue | None]], None]
    ] = field(default_factory=list, kw_only=True)
    """Callbacks that are called when an operation is replaced."""

    def handle_operation_removal(self, op: Operation) -> None:
        """Pass the operation that will be removed to the registered callbacks."""
        for handler in self.operation_removal_handler:
            handler(op)

    def handle_operation_modification(self, op: Operation) -> None:
        """Pass the operation that was just modified to the registered callbacks."""
        for handler in self.operation_modification_handler:
            handler(op)

    def handle_operation_replacement(
        self, op: Operation, new_results: Sequence[SSAValue | None]
    ) -> None:
        """Pass the operation that will be replaced to the registered callbacks."""
        for handler in self.operation_replacement_handler:
            handler(op, new_results)

    def extend_from_listener(self, listener: BuilderListener | PatternRewriterListener):
        """Forward all callbacks from `listener` to this listener."""
        super().extend_from_listener(listener)
        if isinstance(listener, PatternRewriterListener):
            self.operation_removal_handler.extend(listener.operation_removal_handler)
            self.operation_modification_handler.extend(
                listener.operation_modification_handler
            )
            self.operation_replacement_handler.extend(
                listener.operation_replacement_handler
            )


@dataclass(eq=False)
class PatternRewriter(PatternRewriterListener):
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
        if region is self.current_operation.parent_region():
            return True
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
        for op_ in op:
            self.handle_operation_insertion(op_)

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
        for op_ in op:
            self.handle_operation_insertion(op_)

    def insert_op_at_end(self, op: Operation | Sequence[Operation], block: Block):
        """Insert operations in a block contained in the matched operation."""
        if not self._can_modify_block(block):
            raise Exception("Cannot insert operations in block.")
        self.has_done_action = True
        op = [op] if isinstance(op, Operation) else op
        if len(op) == 0:
            return
        block.add_ops(op)
        for op_ in op:
            self.handle_operation_insertion(op_)

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
        for op_ in op:
            self.handle_operation_insertion(op_)
        if target_op is self.current_operation:
            self.added_operations_before.extend(op)

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
        for op_ in ops:
            self.handle_operation_insertion(op_)
        if target_op is self.current_operation:
            self.added_operations_after.extend(ops)

    def erase_matched_op(self, safe_erase: bool = True):
        """
        Erase the operation that was matched to.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
        self.has_erased_matched_operation = True
        self.handle_operation_removal(self.current_operation)
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
        self.handle_operation_removal(op)
        Rewriter.erase_op(op, safe_erase=safe_erase)

    def _replace_all_uses_with(
        self, from_: SSAValue, to: SSAValue | None, safe_erase: bool = True
    ):
        """Replace all uses of an SSA value with another SSA value."""
        for use in from_.uses:
            self.handle_operation_modification(use.operation)
        if to is None:
            from_.erase(safe_erase=safe_erase)
        else:
            from_.replace_by(to)

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
        if not self._can_modify_op(op):
            raise Exception(
                "PatternRewriter can only replace operations that are the matched "
                "operation, or that are contained in the matched operation."
            )
        if isinstance(new_ops, Operation):
            new_ops = [new_ops]

        # First, insert the new operations before the matched operation
        self.insert_op_before(new_ops, op)

        if isinstance(new_ops, Operation):
            new_ops = [new_ops]
        if new_results is None:
            new_results = [] if len(new_ops) == 0 else new_ops[-1].results

        if len(op.results) != len(new_results):
            raise ValueError(
                f"Expected {len(op.results)} new results, but got {len(new_results)}"
            )

        # Then, replace the results with new ones
        self.handle_operation_replacement(op, new_results)
        for old_result, new_result in zip(op.results, new_results):
            self._replace_all_uses_with(old_result, new_result)

        if op.results:
            for new_op in new_ops:
                for res in new_op.results:
                    res.name_hint = op.results[0].name_hint

        # Then, erase the original operation
        self.erase_op(op, safe_erase=safe_erase)

    def modify_block_argument_type(self, arg: BlockArgument, new_type: Attribute):
        """
        Modify the type of a block argument.
        The block should be contained in the matched operation.
        """
        if not self._can_modify_block(arg.block):
            raise ValueError(
                "Cannot modify blocks that are not contained in the matched operation"
            )
        self.has_done_action = True
        arg.type = new_type

        for use in arg.uses:
            self.handle_operation_modification(use.operation)

    def insert_block_argument(
        self, block: Block, index: int, arg_type: Attribute
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
        return block.insert_arg(arg_type, index)

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
        self._replace_all_uses_with(arg, None, safe_erase=safe_erase)
        arg.block.erase_arg(arg, safe_erase)

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

    def inline_region_before(self, region: Region, target: Block) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        if not self._can_modify_region(region):
            raise Exception(
                "Cannot move regions that are not children of the matched operation"
            )
        if not self._can_modify_block(target):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        Rewriter.inline_region_before(region, target)

    def inline_region_after(self, region: Region, target: Block) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        if not self._can_modify_region(region):
            raise Exception(
                "Cannot move regions that are not children of the matched operation"
            )
        if not self._can_modify_block(target):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        Rewriter.inline_region_after(region, target)

    def inline_region_at_start(self, region: Region, target: Region) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        if not self._can_modify_region(region):
            raise Exception(
                "Cannot move regions that are not children of the matched operation"
            )
        if not self._can_modify_region(target):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        Rewriter.inline_region_at_start(region, target)

    def inline_region_at_end(self, region: Region, target: Region) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        if not self._can_modify_region(region):
            raise Exception(
                "Cannot move regions that are not children of the matched operation"
            )
        if not self._can_modify_region(target):
            raise Exception(
                "Cannot move blocks that are not contained in the matched operation."
            )
        Rewriter.inline_region_at_end(region, target)

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


@dataclass
class TypeConversionPattern(RewritePattern):
    """
    Base pattern for type conversion. It is supposed to be inherited from, then one can
    implement `convert_type` to define the conversion.

    It will convert an Operations' result types, dictionary attributes, and block arguments.

    One can use `@attr_type_rewrite_pattern` on this defined method to automatically filter
    on the Attribute type used.

    This base pattern defines two flags:

    - `recursive` (defaulting to False): recurse over structured attributes to convert
      parameters.
      e.g. a recusrive `i32` to `index` conversion will convert `vector<i32>` to
      `vector<index>`.
    - `ops` (defaulting to any Operation) is a tuple of Operation types on which to apply
      the defined attribute conversion.
    """

    recursive: bool = False
    """
    recurse over structured attributes to convert parameters.
    Defaults to False.
    """
    ops: tuple[type[Operation], ...] | None = None
    """
    A tuple of Operation types on which to apply the defined attribute conversion.
    Defaults to any operation type.
    """

    @abstractmethod
    def convert_type(self, typ: Attribute, /) -> Attribute | None:
        """
        The method to implement to define a TypeConversionPattern

        This defines how the input Attribute should be converted.
        It allows returning None, meaning "this attribute should not
        be converted".
        """
        raise NotImplementedError()

    @final
    def _convert_type_rec(self, typ: Attribute) -> Attribute | None:
        """
        Provided recursion over structed/parameterized Attributes.
        """
        inp = typ
        if self.recursive:
            if isinstance(typ, ParametrizedAttribute):
                parameters = list(
                    self._convert_type_rec(p) or p for p in typ.parameters
                )
                inp = type(typ).new(parameters)
            if isa(typ, ArrayAttr[Attribute]):
                parameters = tuple(self._convert_type_rec(p) or p for p in typ)
                inp = type(typ).new(parameters)
        converted = self.convert_type(inp)
        return converted if converted is not None else inp

    @final
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        """
        Pattern application implementation
        """
        if self.ops and not isinstance(op, self.ops):
            return
        new_result_types: list[Attribute] = []
        new_attributes: dict[str, Attribute] = {}
        new_properties: dict[str, Attribute] = {}
        changed: bool = False
        for result in op.results:
            converted = self._convert_type_rec(result.type)
            new_result_types.append(converted or result.type)
            if converted is not None and converted != result.type:
                changed = True
        for name, attribute in op.attributes.items():
            converted = self._convert_type_rec(attribute)
            new_attributes[name] = converted or attribute
            if converted is not None and converted != attribute:
                changed = True
        for name, attribute in op.properties.items():
            converted = self._convert_type_rec(attribute)
            new_properties[name] = converted or attribute
            if converted is not None and converted != attribute:
                changed = True
        for region in op.regions:
            for block in region.blocks:
                for arg in block.args:
                    converted = self.convert_type(arg.type)
                    if converted is not None and converted != arg.type:
                        rewriter.modify_block_argument_type(arg, converted)
        if changed:
            regions = [op.detach_region(r) for r in op.regions]
            new_op = type(op).create(
                operands=op.operands,
                result_types=new_result_types,
                properties=new_properties,
                attributes=new_attributes,
                successors=op.successors,
                regions=regions,
            )
            rewriter.replace_matched_op(new_op)
            for new, old in zip(new_op.results, op.results):
                new.name_hint = old.name_hint


_TypeConversionPatternT = TypeVar(
    "_TypeConversionPatternT", bound=TypeConversionPattern
)
_AttributeT = TypeVar("_AttributeT", bound=Attribute)
_ConvertedT = TypeVar("_ConvertedT", bound=Attribute)


def attr_type_rewrite_pattern(
    func: Callable[[_TypeConversionPatternT, _AttributeT], _ConvertedT]
) -> Callable[[_TypeConversionPatternT, Attribute], Attribute | None]:
    """
    This function is intended to be used as a decorator on a TypeConversionPattern
    method. It uses type hints to match on a specific attribute type before
    calling the decorated function.
    """
    params = list(inspect.signature(func).parameters.values())
    expected_type: type[_AttributeT] = params[-1].annotation

    @wraps(func)
    def impl(self: _TypeConversionPatternT, typ: Attribute) -> Attribute | None:
        if isa(typ, expected_type):
            return func(self, typ)
        return None

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


@dataclass(eq=False)
class Worklist:
    _op_stack: list[Operation | None] = field(default_factory=list, init=False)
    """
    The list of operations to iterate over, used as a last-in-first-out stack.
    Operations are added and removed at the end of the list.
    Operation that are `None` are meant to be discarded, and are used to
    keep removal of operations O(1).
    """

    _map: dict[Operation, int] = field(default_factory=dict, init=False)
    """
    The map of operations to their index in the stack.
    It is used to check if an operation is already in the stack, and to
    remove it in O(1).
    """

    def is_empty(self) -> bool:
        """Check if the worklist is empty."""
        while self._op_stack and self._op_stack[-1] is None:
            self._op_stack.pop()
        return not bool(self._op_stack)

    def push(self, op: Operation):
        """
        Push an operation to the end of the worklist, if it is not already in it.
        """
        if op not in self._map:
            self._map[op] = len(self._op_stack)
            self._op_stack.append(op)

    def pop(self) -> Operation | None:
        """Pop the operation at the end of the worklist."""
        # All `None` operations at the end of the stack are discarded,
        # as they were removed previously.
        # We either return `None` if the stack is empty, or the last operation
        # that is not `None`.
        while self._op_stack:
            op = self._op_stack.pop()
            if op is not None:
                del self._map[op]
                return op
        return None

    def remove(self, op: Operation):
        """Remove an operation from the worklist."""
        if op in self._map:
            index = self._map[op]
            self._op_stack[index] = None
            del self._map[op]


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

    listener: PatternRewriterListener = field(default_factory=PatternRewriterListener)
    """The listener that will be called when an operation or block is modified."""

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
        rewriter.extend_from_listener(self.listener)
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
