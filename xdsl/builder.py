from __future__ import annotations

from typing import ParamSpec, Callable, TypeAlias, overload

from dataclasses import dataclass

from xdsl.ir import Operation, OperationInvT, Attribute, Region, Block, BlockArgument
from xdsl.dialects.builtin import FunctionType

_P = ParamSpec('_P')


@dataclass
class Builder:
    """
    A helper class to construct IRs, by keeping track of where to insert an operation.
    Currently can only append an operation to a given block, in the future will mirror the
    API of `OpBuilder` in MLIR.

    https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder.html
    """

    block: Block

    def add_op(self, op: Operation):
        self.block.add_op(op)

    def create(self, func: Callable[_P, OperationInvT], *args: _P.args,
               **kwargs: _P.kwargs) -> OperationInvT:
        op = func(*args, **kwargs)
        self.add_op(op)
        return op

    @staticmethod
    def region(func: Callable[[Builder], None]) -> Region:

        block = Block()
        builder = Builder(block)

        func(builder)

        return Region.from_block_list([block])

    @staticmethod
    def _callable_region_args(
        types: tuple[list[Attribute], list[Attribute]]
    ) -> Callable[[_CallableRegionFuncType], tuple[Region, FunctionType]]:

        input_types, return_types = types

        def wrapper(
                func: _CallableRegionFuncType) -> tuple[Region, FunctionType]:

            block = Block.from_arg_types(input_types)
            builder = Builder(block)

            func(builder, block.args)

            region = Region.from_block_list([block])
            ftype = FunctionType.from_lists(input_types, return_types)
            return region, ftype

        return wrapper

    @staticmethod
    def _callable_region_no_args(
            func: Callable[[Builder], None]) -> tuple[Region, FunctionType]:

        @Builder._callable_region_args(([], []))
        def res(builder: Builder, args: tuple[BlockArgument, ...]) -> None:
            func(builder)

        return res

    @overload
    @staticmethod
    def callable_region(
        input: tuple[list[Attribute], list[Attribute]]
    ) -> Callable[[_CallableRegionFuncType], tuple[Region, FunctionType]]:
        ...

    @overload
    @staticmethod
    def callable_region(
            input: Callable[[Builder], None]) -> tuple[Region, FunctionType]:
        ...

    @staticmethod
    def callable_region(
        input: tuple[list[Attribute], list[Attribute]]
        | Callable[[Builder], None]
    ) -> Callable[[_CallableRegionFuncType],
                  tuple[Region, FunctionType]] | tuple[Region, FunctionType]:
        """
        Annotation used to construct a (Region, FunctionType) tuple from a function.
        The annotation can be used in two ways:

        For regions that have inputs or outputs:
        ```
        @Builder.callable_region((input_types, output_types))
        def func(builder: Builder, args: tuple[BlockArgument, ...]) -> None:
            ...
        ```

        For regions that don't have inputs or outputs:
        ``` python
        @Builder.callable_region
        def func(builder: Builder) -> None:
            ...
        ```
        """
        if isinstance(input, tuple):
            return Builder._callable_region_args(input)
        else:
            return Builder._callable_region_no_args(input)


_CallableRegionFuncType: TypeAlias = Callable[
    [Builder, tuple[BlockArgument, ...]], None]
