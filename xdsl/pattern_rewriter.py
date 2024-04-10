from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import wraps
from types import UnionType
from typing import TypeVar, Union, final, get_args, get_origin

from xdsl.builder import BuilderListener, InsertPoint
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, ModuleOp
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    ErasedSSAValue,
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
        default_factory=list, kw_only=True
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

    has_done_action: bool = field(default=False, init=False)
    """Has the rewriter done any action during the current match."""

    def insert_op_at_location(
        self, op: Operation | Sequence[Operation], insertion_point: InsertPoint
    ):
        """Insert operations at a certain location in a block."""
        self.has_done_action = True
        op = (op,) if isinstance(op, Operation) else op
        if not op:
            return
        Rewriter.insert_ops_at_location(op, insertion_point)

        for op_ in op:
            self.handle_operation_insertion(op_)

    def insert_op_before_matched_op(self, op: Operation | Sequence[Operation]):
        """Insert operations before the matched operation."""
        self.insert_op_at_location(op, InsertPoint.before(self.current_operation))

    def insert_op_after_matched_op(self, op: Operation | Sequence[Operation]):
        """Insert operations after the matched operation."""
        self.insert_op_at_location(op, InsertPoint.after(self.current_operation))

    def insert_op_at_end(self, op: Operation | Sequence[Operation], block: Block):
        """Insert operations at the end of a block."""
        self.insert_op_at_location(op, InsertPoint.at_end(block))

    def insert_op_at_start(self, op: Operation | Sequence[Operation], block: Block):
        """Insert operations at the start of a block."""
        self.insert_op_at_location(op, InsertPoint.at_start(block))

    def insert_op_before(
        self, op: Operation | Sequence[Operation], target_op: Operation
    ):
        """Insert operations before an operation."""
        self.insert_op_at_location(op, InsertPoint.before(target_op))

    def insert_op_after(
        self, op: Operation | Sequence[Operation], target_op: Operation
    ):
        """Insert operations after an operation."""
        self.insert_op_at_location(op, InsertPoint.after(target_op))

    def erase_matched_op(self, safe_erase: bool = True):
        """
        Erase the operation that was matched to.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.erase_op(self.current_operation, safe_erase=safe_erase)

    def erase_op(self, op: Operation, safe_erase: bool = True):
        """
        Erase an operation.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
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
        Also, optionally specify SSA values to replace the operation results.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
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
        """Modify the type of a block argument."""
        self.has_done_action = True
        arg.type = new_type

        for use in arg.uses:
            self.handle_operation_modification(use.operation)

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
        self._replace_all_uses_with(arg, None, safe_erase=safe_erase)
        arg.block.erase_arg(arg, safe_erase)

    def inline_block_at_end(self, block: Block, target_block: Block):
        """
        Move the block operations to the end of another block.
        This block should not be a parent of the block to move to.
        """
        self.has_done_action = True
        Rewriter.inline_block_at_end(block, target_block)

    def inline_block_at_start(self, block: Block, target_block: Block):
        """
        Move the block operations to the start of another block.
        This block should not be a parent of the block to move to.
        """
        self.has_done_action = True
        Rewriter.inline_block_at_start(block, target_block)

    def inline_block_before_matched_op(self, block: Block):
        """
        Move the block operations before the matched operation.
        The block should not be a parent of the operation.
        """
        self.inline_block_before(block, self.current_operation)

    def inline_block_before(
        self, block: Block, op: Operation, arg_values: Sequence[SSAValue] = ()
    ):
        """
        Move the block operations before the given operation.
        The block should not be a parent of the operation.
        """
        self.has_done_action = True
        Rewriter.inline_block_before(block, op, arg_values=arg_values)

    def inline_block_after_matched_op(self, block: Block):
        """
        Move the block operations after the matched operation.
        The block should not be a parent of the operation.
        """
        self.inline_block_after(block, self.current_operation)

    def inline_block_after(self, block: Block, op: Operation):
        """
        Move the block operations after the given operation.
        The block should not be a parent of the operation.
        """
        self.has_done_action = True
        Rewriter.inline_block_after(block, op)

    def move_region_contents_to_new_regions(self, region: Region) -> Region:
        """Move the region blocks to a new region."""
        self.has_done_action = True
        return Rewriter.move_region_contents_to_new_regions(region)

    def inline_region_before(self, region: Region, target: Block) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        Rewriter.inline_region_before(region, target)

    def inline_region_after(self, region: Region, target: Block) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        Rewriter.inline_region_after(region, target)

    def inline_region_at_start(self, region: Region, target: Region) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        Rewriter.inline_region_at_start(region, target)

    def inline_region_at_end(self, region: Region, target: Region) -> None:
        """Move the region blocks to an existing region."""
        self.has_done_action = True
        Rewriter.inline_region_at_end(region, target)


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
            if isa(typ, DictionaryAttr):
                parameters = {k: self._convert_type_rec(v) for k, v in typ.data.items()}
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
    func: Callable[[_TypeConversionPatternT, _AttributeT], _ConvertedT | None]
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

    _worklist: Worklist = field(default_factory=Worklist, init=False)
    """The worklist of operations to walk over."""

    def _add_operands_to_worklist(self, operands: Iterable[SSAValue]) -> None:
        """
        Add defining operations of SSA values to the worklist if they have only
        one use. This is a heuristic based on the fact that single-use operations
        have more canonicalization opportunities.
        """
        for operand in operands:
            if (
                len(operand.uses) == 1
                and not isinstance(operand, ErasedSSAValue)
                and isinstance((op := operand.owner), Operation)
            ):
                self._worklist.push(op)

    def _handle_operation_insertion(self, op: Operation) -> None:
        """Handle insertion of an operation."""
        if self.apply_recursively:
            self._worklist.push(op)

    def _handle_operation_removal(self, op: Operation) -> None:
        """Handle removal of an operation."""
        if self.apply_recursively:
            self._add_operands_to_worklist(op.operands)
        self._worklist.remove(op)

    def _handle_operation_modification(self, op: Operation) -> None:
        """Handle modification of an operation."""
        if self.apply_recursively:
            self._worklist.push(op)

    def _handle_operation_replacement(
        self, op: Operation, new_results: Sequence[SSAValue | None]
    ) -> None:
        """Handle replacement of an operation."""
        if self.apply_recursively:
            for result in op.results:
                for user in result.uses:
                    self._worklist.push(user.operation)

    def _get_rewriter_listener(self) -> PatternRewriterListener:
        """
        Get the listener that will be passed to the rewriter.
        It will take care of adding operations to the worklist, and calling the
        listener passed as configuration to the walker.
        """
        return PatternRewriterListener(
            operation_insertion_handler=[
                *self.listener.operation_insertion_handler,
                self._handle_operation_insertion,
            ],
            operation_removal_handler=[
                *self.listener.operation_removal_handler,
                self._handle_operation_removal,
            ],
            operation_modification_handler=[
                *self.listener.operation_modification_handler,
                self._handle_operation_modification,
            ],
            operation_replacement_handler=[
                *self.listener.operation_replacement_handler,
                self._handle_operation_replacement,
            ],
            block_creation_handler=self.listener.block_creation_handler,
        )

    def rewrite_module(self, module: ModuleOp) -> bool:
        """
        Rewrite operations nested in the given operation by repeatedly applying the
        pattern. Returns `True` if the IR was mutated.
        """
        return self.rewrite_op(module)

    def rewrite_op(self, op: Operation) -> bool:
        """
        Rewrite operations nested in the given operation by repeatedly applying the
        pattern. Returns `True` if the IR was mutated.
        """
        pattern_listener = self._get_rewriter_listener()

        self._populate_worklist(op)
        op_was_modified = self._process_worklist(pattern_listener)

        if not self.apply_recursively:
            return op_was_modified

        result = op_was_modified

        while op_was_modified:
            self._populate_worklist(op)
            op_was_modified = self._process_worklist(pattern_listener)

        return result

    def _populate_worklist(self, op: Operation) -> None:
        """Populate the worklist with all nested operations."""
        # We walk in reverse order since we use a stack for our worklist.
        for sub_op in op.walk(
            reverse=not self.walk_reverse, region_first=not self.walk_regions_first
        ):
            self._worklist.push(sub_op)

    def _process_worklist(self, listener: PatternRewriterListener) -> bool:
        """
        Process the worklist until it is empty.
        Returns true if any modification was done.
        """
        rewriter_has_done_action = False

        # Handle empty worklist
        op = self._worklist.pop()
        if op is None:
            return rewriter_has_done_action

        # Create a rewriter on the first operation
        rewriter = PatternRewriter(op)
        rewriter.extend_from_listener(listener)

        # do/while loop
        while True:
            # Reset the rewriter on `op`
            rewriter.has_done_action = False
            rewriter.current_operation = op

            # Apply the pattern on the operation
            try:
                self.pattern.match_and_rewrite(op, rewriter)
            except Exception as err:
                op.emit_error(
                    f"Error while applying pattern: {str(err)}",
                    exception_type=type(err),
                    underlying_error=err,
                )
            rewriter_has_done_action |= rewriter.has_done_action

            # If the worklist is empty, we are done
            op = self._worklist.pop()
            if op is None:
                return rewriter_has_done_action
