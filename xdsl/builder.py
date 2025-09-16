from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, overload

from typing_extensions import TypeVar

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    OperationInvT,
    Region,
    SSAValue,
)
from xdsl.rewriter import BlockInsertPoint, InsertPoint, Rewriter


@dataclass(eq=False)
class BuilderListener:
    """A listener for builder events."""

    operation_insertion_handler: list[Callable[[Operation], None]] = field(
        default_factory=list[Callable[[Operation], None]], kw_only=True
    )
    """Callbacks that are called when an operation is inserted by the builder."""

    block_creation_handler: list[Callable[[Block], None]] = field(
        default_factory=list[Callable[[Block], None]], kw_only=True
    )
    """Callback that are called when a block is created by the builder."""

    def handle_operation_insertion(self, op: Operation) -> None:
        """Pass the operation that was just inserted to callbacks."""
        for callback in self.operation_insertion_handler:
            callback(op)

    def handle_block_creation(self, block: Block) -> None:
        """Pass the block that was just created to callbacks."""
        for callback in self.block_creation_handler:
            callback(block)

    def extend_from_listener(self, listener: BuilderListener) -> None:
        """Forward all callbacks from `listener` to this listener."""
        self.operation_insertion_handler.extend(listener.operation_insertion_handler)
        self.block_creation_handler.extend(listener.block_creation_handler)


InsertOpInvT = TypeVar("InsertOpInvT", bound=Operation | Sequence[Operation])


@dataclass
class Builder(BuilderListener):
    """
    A helper class to construct IRs, by keeping track of where to insert an
    operation. It mimics the OpBuilder class from MLIR.

    See external [documentation](https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html).
    """

    insertion_point: InsertPoint
    """Operations will be inserted at this location."""

    _name_hint: str | None = field(default=None)
    """
    If this is not None, results of inserted operations will be assigned this name hint.
    This is always valid.
    """

    context: Context | None = field(default=None)
    """
    The context used to load dialects.
    """

    @property
    def name_hint(self) -> str | None:
        return self._name_hint

    @name_hint.setter
    def name_hint(self, name: str | None):
        self._name_hint = SSAValue.extract_valid_name(name)

    def insert(self, op: OperationInvT) -> OperationInvT:
        """
        Inserts op at the current location and returns it.
        """
        return self.insert_op(op)

    def insert_op(
        self,
        op: InsertOpInvT,
        insertion_point: InsertPoint | None = None,
    ) -> InsertOpInvT:
        """Inserts op(s) at the current insertion point."""
        ops = (op,) if isinstance(op, Operation) else op
        if not ops:
            return ops

        implicit_builder = _current_builder.builder
        if implicit_builder is not None and implicit_builder is not self:
            raise ValueError(
                "Cannot insert operation explicitly when an implicit builder exists."
            )

        Rewriter.insert_op(
            op, self.insertion_point if insertion_point is None else insertion_point
        )

        for op_ in ops:
            if self._name_hint is not None:
                for result in op_.results:
                    if result.name_hint is None:
                        result._name = self._name_hint  # pyright: ignore[reportPrivateUsage]
            self.handle_operation_insertion(op_)

        return op

    def create_block(
        self, insert_point: BlockInsertPoint, arg_types: Iterable[Attribute] = ()
    ) -> Block:
        """
        Create a block at the given location, and set the operation insertion point
        at the end of the inserted block.
        """
        block = Block(arg_types=arg_types)
        Rewriter.insert_block(block, insert_point)

        self.insertion_point = InsertPoint.at_end(block)

        self.handle_block_creation(block)
        return block

    @staticmethod
    def _region_no_args(
        func: Callable[[Builder], None], ctx: Context | None = None
    ) -> Region:
        """
        Generates a single-block region.
        """
        block = Block()
        builder = Builder(InsertPoint.at_end(block), context=ctx)
        func(builder)
        return Region(block)

    @staticmethod
    def _region_args(
        input_types: Sequence[Attribute] | ArrayAttr[Attribute],
        ctx: Context | None = None,
    ) -> Callable[[_CallableRegionFuncType], Region]:
        """
        Decorator for constructing a single-block region, containing the implementation of a
        region with some input arguments.
        """

        if isinstance(input_types, ArrayAttr):
            input_types = input_types.data

        def wrapper(func: _CallableRegionFuncType) -> Region:
            block = Block(arg_types=input_types)
            builder = Builder(InsertPoint.at_start(block), context=ctx)

            func(builder, block.args)

            region = Region(block)
            return region

        return wrapper

    @overload
    @staticmethod
    def region(
        input: Sequence[Attribute] | ArrayAttr[Attribute],
        ctx: Context | None = None,
    ) -> Callable[[_CallableRegionFuncType], Region]:
        """
        Annotation used to construct a Region tuple from a function.
        The annotation can be used in two ways:

        For regions that have inputs or outputs:
        ```
        @Builder.region(input_types)
        def func(builder: Builder, args: tuple[BlockArgument, ...]) -> None:
            ...
        ```

        For regions that don't have inputs or outputs:
        ``` python
        @Builder.region
        def func(builder: Builder) -> None:
            ...
        ```
        """
        ...

    @overload
    @staticmethod
    def region(
        input: Callable[[Builder], None], ctx: Context | None = None
    ) -> Region: ...

    @staticmethod
    def region(
        input: Sequence[Attribute] | ArrayAttr[Attribute] | Callable[[Builder], None],
        ctx: Context | None = None,
    ) -> Callable[[_CallableRegionFuncType], Region] | Region:
        if isinstance(input, Callable):
            return Builder._region_no_args(input, ctx)
        else:
            return Builder._region_args(input, ctx)

    @staticmethod
    def _implicit_region_no_args(
        func: Callable[[], None], ctx: Context | None = None
    ) -> Region:
        """
        Generates a single-block region.
        """
        block = Block()
        builder = Builder(InsertPoint.at_end(block))

        with ImplicitBuilder(builder):
            func()

        return Region(block)

    @staticmethod
    def _implicit_region_args(
        input_types: Sequence[Attribute] | ArrayAttr[Attribute],
        ctx: Context | None = None,
    ) -> Callable[[_CallableImplicitRegionFuncType], Region]:
        """
        Decorator for constructing a single-block region, containing the implementation of a
        region with some input arguments.
        """

        if isinstance(input_types, ArrayAttr):
            input_types = input_types.data

        def wrapper(func: _CallableImplicitRegionFuncType) -> Region:
            block = Block(arg_types=input_types)
            builder = Builder(InsertPoint.at_end(block), context=ctx)

            with ImplicitBuilder(builder):
                func(block.args)

            region = Region(block)
            return region

        return wrapper

    @overload
    @staticmethod
    def implicit_region(
        input: Sequence[Attribute] | ArrayAttr[Attribute],
        ctx: Context | None = None,
    ) -> Callable[[_CallableImplicitRegionFuncType], Region]:
        """
        Annotation used to construct a Region tuple from a function.
        The annotation can be used in two ways:

        For regions that have inputs or outputs:
        ```
        @Builder.implicit_region(input_types)
        def func(args: tuple[BlockArgument, ...]) -> None:
            ...
        ```

        For regions that don't have inputs or outputs:
        ``` python
        @Builder.implicit_region
        def func() -> None:
            ...
        ```
        """
        ...

    @overload
    @staticmethod
    def implicit_region(
        input: Callable[[], None], ctx: Context | None = None
    ) -> Region: ...

    @staticmethod
    def implicit_region(
        input: Sequence[Attribute] | ArrayAttr[Attribute] | Callable[[], None],
        ctx: Context | None = None,
    ) -> Callable[[_CallableImplicitRegionFuncType], Region] | Region:
        if isinstance(input, Callable):
            return Builder._implicit_region_no_args(input, ctx)
        else:
            return Builder._implicit_region_args(input, ctx)

    @staticmethod
    def assert_implicit():
        if _current_builder.builder is None:
            raise ValueError(
                "op_builder must be called within an implicit builder block"
            )


# Implicit builders


@dataclass
class _ThreadLocalBuilder(threading.local):
    """
    Stores the implicit builder for use in ImplicitBuilder, None by default.
    There is a builder per thread, guaranteed by inheriting from `threading.local`.
    """

    builder: Builder | None = None


_current_builder = _ThreadLocalBuilder()


@contextlib.contextmanager
def ImplicitBuilder(
    arg: Builder | Block | Region | None,
    ctx: Context | None = None,
):
    """
    Context manager for managing the current implicit builder context, consisting of the stack of builders in
    the current thread, and the current builder.

    Operations created within a `with` block of an implicit builder will be added to it.
    If there are nested implicit builder blocks, the operation will be added to the
    innermost one. Operations cannot be added to multiple blocks, and any attempt to do so
    will result in an exception.

    Example:

    ``` python
    from xdsl.dialects import arith

    with ImplicitBuilder(block):
        arith.Constant(IntegerAttr(5, 32))

    assert len(block.ops) == 1
    assert isinstance(block.ops.first, arith.Constant)
    ```
    """
    match arg:
        case None:
            # None option added as convenience to allow for extending optional regions
            # in ops easily.
            raise ValueError("Cannot pass None to implicit_builder")
        case Region():
            builder = Builder(InsertPoint.at_end(arg.block), context=ctx)
        case Block():
            builder = Builder(InsertPoint.at_end(arg), context=ctx)
        case Builder():
            if ctx is not None:
                raise ValueError(
                    "Cannot pass both a Builder and a Context to implicit_builder"
                )
            builder = arg

    old_builder = _current_builder.builder
    old_post_init = None
    if old_builder is None:
        old_post_init = _override_operation_post_init()

    _current_builder.builder = builder
    try:
        yield builder.insertion_point.block.args
    finally:
        _current_builder.builder = old_builder
        if old_builder is None:
            Operation.__post_init__ = old_post_init  # pyright: ignore[reportAttributeAccessIssue]


_CallableRegionFuncType: TypeAlias = Callable[
    [Builder, tuple[BlockArgument, ...]], None
]
_CallableImplicitRegionFuncType: TypeAlias = Callable[[tuple[BlockArgument, ...]], None]


def _op_init_callback(op: Operation):
    if (b := _current_builder.builder) is not None:
        b.insert(op)


def _override_operation_post_init() -> Callable[[Operation], None]:
    old_post_init = Operation.__post_init__

    def new_post_init(self: Operation) -> None:
        old_post_init(self)
        _op_init_callback(self)

    Operation.__post_init__ = new_post_init
    return old_post_init
