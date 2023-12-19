from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import ClassVar, TypeAlias, overload

from xdsl.dialects.builtin import ArrayAttr
from xdsl.ir import Attribute, Block, BlockArgument, Operation, OperationInvT, Region


@dataclass
class InsertPoint:
    """
    An insert point.
    It is either a point before an operation, or at the end of a block.

    https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder_1_1InsertPoint.html
    """

    block: Block
    """The block where the insertion point is in."""

    insert_before: Operation | None = field(default=None)
    """
    The insertion point is right before this operation.
    If the operation is None, the insertion point is at the end of the block.
    """

    @staticmethod
    def before(op: Operation) -> InsertPoint:
        """Gets the insertion point before an operation."""
        if (block := op.parent_block()) is None:
            raise ValueError("Operation insertion point must have a parent block")
        return InsertPoint(block, op)

    @staticmethod
    def after(op: Operation) -> InsertPoint:
        """Gets the insertion point after an operation."""
        block = op.parent_block()
        if block is None:
            raise ValueError("Operation insertion point must have a parent block")
        return InsertPoint(block, op.next_op)

    @staticmethod
    def at_start(block: Block) -> InsertPoint:
        """Gets the insertion point at the start of a block."""
        return InsertPoint(block, block.ops.first)

    @staticmethod
    def at_end(block: Block) -> InsertPoint:
        """Gets the insertion point at the end of a block."""
        return InsertPoint(block)

    def verify(self) -> None:
        """
        Check that the insertion point is valid.
        An insertion point can only be invalid if `op_before` is an `Operation`, and its
        parent is not `block`.
        """
        if self.insert_before is not None:
            if self.insert_before.parent is not self.block:
                raise ValueError("Insertion point must be in the builder's `block`")


@dataclass
class Builder:
    """
    A helper class to construct IRs, by keeping track of where to insert an
    operation. It mimics the OpBuilder class from MLIR.

    https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    """

    _insertion_point: InsertPoint
    """Operations will be inserted at this location."""

    def __init__(self, insert_point: InsertPoint) -> None:
        self.insertion_point = insert_point

    @staticmethod
    def before(op: Operation) -> Builder:
        """Creates a builder with the insertion point before an operation."""
        return Builder(InsertPoint.before(op))

    @staticmethod
    def after(op: Operation) -> Builder:
        """Creates a builder with the insertion point after an operation."""
        return Builder(InsertPoint.after(op))

    @staticmethod
    def at_start(block: Block) -> Builder:
        """Creates a builder with the insertion point at the start of a block."""
        return Builder(InsertPoint.at_start(block))

    @staticmethod
    def at_end(block: Block) -> Builder:
        """Creates a builder with the insertion point at the end of a block."""
        return Builder(InsertPoint.at_end(block))

    @property
    def insertion_point(self) -> InsertPoint:
        return self._insertion_point

    @insertion_point.setter
    def insertion_point(self, insertion_point: InsertPoint):
        self._insertion_point = insertion_point
        self._insertion_point.verify()

    def insert(self, op: OperationInvT) -> OperationInvT:
        """Inserts `op` at the current insertion point."""

        implicit_builder = ImplicitBuilder.get()

        if implicit_builder is not None and implicit_builder is not self:
            raise ValueError(
                "Cannot insert operation explicitly when an implicit " "builder exists."
            )

        block = self.insertion_point.block
        insert_before = self.insertion_point.insert_before
        if insert_before is not None:
            block.insert_op_before(op, insert_before)
        else:
            block.add_op(op)

        return op

    @staticmethod
    def _region_no_args(func: Callable[[Builder], None]) -> Region:
        """
        Generates a single-block region.
        """
        block = Block()
        builder = Builder.at_end(block)
        func(builder)
        return Region(block)

    @staticmethod
    def _region_args(
        input_types: Sequence[Attribute] | ArrayAttr[Attribute],
    ) -> Callable[[_CallableRegionFuncType], Region]:
        """
        Decorator for constructing a single-block region, containing the implementation of a
        region with some input arguments.
        """

        if isinstance(input_types, ArrayAttr):
            input_types = input_types.data

        def wrapper(func: _CallableRegionFuncType) -> Region:
            block = Block(arg_types=input_types)
            builder = Builder.at_start(block)

            func(builder, block.args)

            region = Region(block)
            return region

        return wrapper

    @overload
    @staticmethod
    def region(
        input: Sequence[Attribute] | ArrayAttr[Attribute],
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
    def region(input: Callable[[Builder], None]) -> Region:
        ...

    @staticmethod
    def region(
        input: Sequence[Attribute] | ArrayAttr[Attribute] | Callable[[Builder], None]
    ) -> Callable[[_CallableRegionFuncType], Region] | Region:
        if isinstance(input, Callable):
            return Builder._region_no_args(input)
        else:
            return Builder._region_args(input)

    @staticmethod
    def _implicit_region_no_args(func: Callable[[], None]) -> Region:
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
    ) -> Callable[[_CallableImplicitRegionFuncType], Region]:
        """
        Decorator for constructing a single-block region, containing the implementation of a
        region with some input arguments.
        """

        if isinstance(input_types, ArrayAttr):
            input_types = input_types.data

        def wrapper(func: _CallableImplicitRegionFuncType) -> Region:
            block = Block(arg_types=input_types)
            builder = Builder(InsertPoint.at_end(block))

            with ImplicitBuilder(builder):
                func(block.args)

            region = Region(block)
            return region

        return wrapper

    @overload
    @staticmethod
    def implicit_region(
        input: Sequence[Attribute] | ArrayAttr[Attribute],
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
    def implicit_region(input: Callable[[], None]) -> Region:
        ...

    @staticmethod
    def implicit_region(
        input: Sequence[Attribute] | ArrayAttr[Attribute] | Callable[[], None]
    ) -> Callable[[_CallableImplicitRegionFuncType], Region] | Region:
        if isinstance(input, Callable):
            return Builder._implicit_region_no_args(input)
        else:
            return Builder._implicit_region_args(input)

    @staticmethod
    def assert_implicit():
        if ImplicitBuilder.get() is None:
            raise ValueError(
                "op_builder must be called within an implicit builder block"
            )


# Implicit builders


@dataclass
class _ImplicitBuilderStack(threading.local):
    """
    Stores the stack of implicit builders for use in @Builder.implicit_region, empty by
    default. There is a stack per thread, guaranteed by inheriting from `threading.local`.
    """

    stack: list[Builder] = field(default_factory=list)

    def push(self, builder: Builder) -> None:
        self.stack.append(builder)

    def get(self) -> Builder | None:
        if len(self.stack):
            return self.stack[-1]

    def pop(self, builder: Builder) -> Builder:
        popped = self.stack.pop()
        assert popped is builder
        return popped


class ImplicitBuilder(contextlib.AbstractContextManager[tuple[BlockArgument, ...]]):
    """
    Stores the current implicit builder context, consisting of the stack of builders in
    the current thread, and the current builder.

    Operations created within a `with` block of an implicit builder will be added to it.
    If there are nested implicit builder blocks, the operation will be added to the
    innermost one. Operations cannot be added to multiple blocks, and any attempt to do so
    will result in an exception.

    Example:

    ``` python
    from xdsl.dialects import arith

    block = Block()
    builder = Builder(block)

    with builder.implicit():
        arith.Constant.from_int_and_width(5, 32)

    assert len(block.ops) == 1
    assert isinstance(block.ops.first, arith.Constant)
    ```
    """

    _stack: ClassVar[_ImplicitBuilderStack] = _ImplicitBuilderStack()

    _builder: Builder

    def __init__(self, arg: Builder | Block | Region | None):
        if arg is None:
            # None option added as convenience to allow for extending optional regions in
            # ops easily
            raise ValueError("Cannot pass None to ImplicitBuidler init")
        if isinstance(arg, Region):
            arg = arg.block
        if isinstance(arg, Block):
            arg = Builder.at_end(arg)
        self._builder = arg

    def __enter__(self) -> tuple[BlockArgument, ...]:
        type(self)._stack.push(self._builder)
        return self._builder.insertion_point.block.args

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        type(self)._stack.pop(self._builder)

    @classmethod
    def get(cls) -> Builder | None:
        """
        Gets the topmost ImplicitBuilder on the stack.
        """
        return cls._stack.get()


_CallableRegionFuncType: TypeAlias = Callable[
    [Builder, tuple[BlockArgument, ...]], None
]
_CallableImplicitRegionFuncType: TypeAlias = Callable[[tuple[BlockArgument, ...]], None]


def _op_init_callback(op: Operation):
    if (b := ImplicitBuilder.get()) is not None:
        b.insert(op)


def _override_operation_post_init() -> None:
    old_post_init = Operation.__post_init__

    def new_post_init(self: Operation) -> None:
        old_post_init(self)
        _op_init_callback(self)

    Operation.__post_init__ = new_post_init


# set up the operation callback for implicit construction
_override_operation_post_init()
