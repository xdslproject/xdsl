from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from types import TracebackType
from typing import Callable, ClassVar, Sequence, TypeAlias, overload

from xdsl.dialects.builtin import ArrayAttr
from xdsl.ir import Attribute, Block, BlockArgument, Operation, OperationInvT, Region


@dataclass
class Builder:
    """
    A helper class to construct IRs, by keeping track of where to insert an
    operation. Currently the insertion point is always at the end of the block.
    In the future will closely follow the API of `OpBuilder` in MLIR, inserting
    at arbitrary locations.

    https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    """

    block: Block
    """
    Operations will be inserted in this block.
    """

    _insertion_point: Operation | None = field(default=None)
    """
    Operations will be inserted before this operation, or at the end of the block if None.
    """

    def __post_init__(self):
        self._verify_insertion_point()

    def _verify_insertion_point(self):
        if self.insertion_point is not None:
            if self.insertion_point.parent is not self.block:
                raise ValueError("Insertion point must be in the builder's `block`")

    @property
    def insertion_point(self) -> Operation | None:
        return self._insertion_point

    @insertion_point.setter
    def insertion_point(self, insertion_point: Operation | None):
        self._insertion_point = insertion_point
        self._verify_insertion_point()

    def insert(self, op: OperationInvT) -> OperationInvT:
        """
        Inserts `op` in `self.block` at the end of the block.
        """

        implicit_builder = ImplicitBuilder.get()

        if implicit_builder is not None and implicit_builder is not self:
            raise ValueError(
                "Cannot insert operation explicitly when an implicit " "builder exists."
            )

        if self.insertion_point is not None:
            self.block.insert_op_before(op, self.insertion_point)
        else:
            self.block.add_op(op)
        return op

    @staticmethod
    def _region_no_args(func: Callable[[Builder], None]) -> Region:
        """
        Generates a single-block region.
        """
        block = Block()
        builder = Builder(block)
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
            builder = Builder(block)

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
        builder = Builder(block)

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
            builder = Builder(block)

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
            arg = Builder(arg)
        self._builder = arg

    def __enter__(self) -> tuple[BlockArgument, ...]:
        type(self)._stack.push(self._builder)
        return self._builder.block.args

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
