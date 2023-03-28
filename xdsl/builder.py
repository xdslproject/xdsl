from __future__ import annotations

from typing import ParamSpec, Callable, TypeAlias, overload

from dataclasses import dataclass

from xdsl.ir import Operation, OperationInvT, Attribute, Region, Block, BlockArgument
from xdsl.dialects.builtin import FunctionType

_P = ParamSpec('_P')


@dataclass
class OpBuilder:
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
    def region(func: Callable[[OpBuilder], None]) -> Region:

        block = Block()
        builder = OpBuilder(block)

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
            builder = OpBuilder(block)

            func(builder, block.args)

            region = Region.from_block_list([block])
            ftype = FunctionType.from_lists(input_types, return_types)
            return region, ftype

        return wrapper

    @staticmethod
    def _callable_region_no_args(
            func: Callable[[OpBuilder], None]) -> tuple[Region, FunctionType]:

        @OpBuilder._callable_region_args(([], []))
        def res(builder: OpBuilder, args: tuple[BlockArgument, ...]) -> None:
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
            input: Callable[[OpBuilder], None]) -> tuple[Region, FunctionType]:
        ...

    @staticmethod
    def callable_region(
        input: tuple[list[Attribute], list[Attribute]]
        | Callable[[OpBuilder], None]
    ) -> Callable[[_CallableRegionFuncType],
                  tuple[Region, FunctionType]] | tuple[Region, FunctionType]:
        """
        Annotation used to construct a (Region, FunctionType) tuple from a function.
        The annotation can be used in two ways:

        For regions that have inputs or outputs:
        ```
        @OpBuilder.callable_region((input_types, output_types))
        def func(builder: OpBuilder, args: tuple[BlockArgument, ...]) -> None:
            ...
        ```

        For regions that don't have inputs or outputs:
        ``` python
        @OpBuilder.callable_region
        def func(builder: OpBuilder) -> None:
            ...
        ```
        """
        if isinstance(input, tuple):
            return OpBuilder._callable_region_args(input)
        else:
            return OpBuilder._callable_region_no_args(input)


_CallableRegionFuncType: TypeAlias = Callable[
    [OpBuilder, tuple[BlockArgument, ...]], None]
