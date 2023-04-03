from __future__ import annotations
from types import TracebackType

from typing import Callable, ClassVar, TypeAlias, overload
import threading
import contextlib

from dataclasses import dataclass, field
from xdsl.dialects.builtin import ArrayAttr

from xdsl.ir import Operation, OperationInvT, Attribute, Region, Block, BlockArgument


@dataclass
class _ImplicitBuilders(threading.local):
    bb: list[Builder] = field(default_factory=list)


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

    _local: ClassVar[_ImplicitBuilders] = _ImplicitBuilders()

    def insert(self, op: OperationInvT) -> OperationInvT:
        """
        Inserts `op` in `self.block` at the end of the block.
        """

        implicit_builder = Builder.get_implicit_builder()

        if implicit_builder is not None and implicit_builder is not self:

            raise ValueError(
                'Cannot insert operation explicitly when an implicit '
                'builder exists.')

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
        return Region.from_block_list([block])

    @staticmethod
    def _region_args(
        input_types: list[Attribute] | ArrayAttr[Attribute]
    ) -> Callable[[_CallableRegionFuncType], Region]:
        """
        Decorator for constructing a single-block region, containing the implementation of a
        function with some input arguments.
        """

        if isinstance(input_types, ArrayAttr):
            input_type_seq = input_types.data
        else:
            input_type_seq = input_types

        def wrapper(func: _CallableRegionFuncType) -> Region:
            block = Block.from_arg_types(input_type_seq)
            builder = Builder(block)

            func(builder, block.args)

            region = Region.from_block_list([block])
            return region

        return wrapper

    @overload
    @staticmethod
    def region(
        input: list[Attribute] | ArrayAttr[Attribute]
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
        input: list[Attribute] | ArrayAttr[Attribute]
        | Callable[[Builder], None]
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

        with Builder._push_implicit_builder(builder):
            func()

        return Region.from_block_list([block])

    @staticmethod
    def _implicit_region_args(
        input_types: list[Attribute] | ArrayAttr[Attribute]
    ) -> Callable[[_CallableImplicitRegionFuncType], Region]:
        """
        Decorator for constructing a single-block region, containing the implementation of a
        function with some input arguments.
        """

        if isinstance(input_types, ArrayAttr):
            input_type_seq = input_types.data
        else:
            input_type_seq = input_types

        def wrapper(func: _CallableImplicitRegionFuncType) -> Region:
            block = Block.from_arg_types(input_type_seq)
            builder = Builder(block)

            with Builder._push_implicit_builder(builder):
                func(block.args)

            region = Region.from_block_list([block])
            return region

        return wrapper

    @overload
    @staticmethod
    def implicit_region(
        input: list[Attribute] | ArrayAttr[Attribute]
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
        def func(builder: Builder) -> None:
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
        input: list[Attribute] | ArrayAttr[Attribute]
        | Callable[[], None]
    ) -> Callable[[_CallableImplicitRegionFuncType], Region] | Region:
        if isinstance(input, Callable):
            return Builder._implicit_region_no_args(input)
        else:
            return Builder._implicit_region_args(input)

    @classmethod
    def _push_implicit_builder(cls, builder: Builder) -> _ImplicitBuilder:
        return _ImplicitBuilder(cls._local.bb, builder)

    @classmethod
    def get_implicit_builder(cls) -> Builder | None:
        bb = cls._local.bb
        if len(bb):
            return bb[-1]


@dataclass
class _ImplicitBuilder(contextlib.AbstractContextManager[None]):

    bb: list[Builder]
    builder: Builder

    def __enter__(self) -> None:
        self.bb.append(self.builder)

    def __exit__(self, __exc_type: type[BaseException] | None,
                 __exc_value: BaseException | None,
                 __traceback: TracebackType | None) -> bool | None:
        popped = self.bb.pop()
        assert popped is self.builder


_CallableRegionFuncType: TypeAlias = Callable[
    [Builder, tuple[BlockArgument, ...]], None]
_CallableImplicitRegionFuncType: TypeAlias = Callable[
    [tuple[BlockArgument, ...]], None]


def _op_init_callback(op: Operation):
    if (b := Builder.get_implicit_builder()) is not None:
        b.insert(op)


def _override_operation_post_init() -> None:
    old_post_init = Operation.__post_init__

    def new_post_init(self: Operation) -> None:
        old_post_init(self)
        _op_init_callback(self)

    Operation.__post_init__ = new_post_init


# set up the operation callback for implicit construction
_override_operation_post_init()
