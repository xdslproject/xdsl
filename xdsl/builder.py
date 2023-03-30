from __future__ import annotations

from typing import Callable, TypeAlias, overload

from dataclasses import dataclass

from xdsl.ir import OperationInvT, Attribute, Region, Block, BlockArgument
from xdsl.dialects.builtin import FunctionType


@dataclass
class OpBuilder:
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

    def insert(self, op: OperationInvT) -> OperationInvT:
        """
        Inserts `op` in `self.block` at the end of the block.
        """

        self.block.add_op(op)
        return op

    @staticmethod
    def region(func: Callable[[OpBuilder], None]) -> Region:
        """
        Generates a single-block region.
        """

        block = Block()
        builder = OpBuilder(block)

        func(builder)

        return Region.from_block_list([block])

    @staticmethod
    def _callable_region_args(
        types: tuple[list[Attribute], list[Attribute]]
    ) -> Callable[[_CallableRegionFuncType], tuple[Region, FunctionType]]:
        """
        Constructs a tuple of (Region, FunctionType). The Region is a 
        single-block region, containing the implementation of a function.
        `types` specifies the input and result types of the function.
        """

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
        """
        Constructs a tuple of (Region, FunctionType) for a region that takes no arguments
        and returns no results.
        """

        @OpBuilder._callable_region_args(([], []))
        def res(builder: OpBuilder, args: tuple[BlockArgument, ...]) -> None:
            func(builder)

        return res

    @overload
    @staticmethod
    def callable_region(
        input: tuple[list[Attribute], list[Attribute]]
    ) -> Callable[[_CallableRegionFuncType], tuple[Region, FunctionType]]:
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
        if isinstance(input, tuple):
            return OpBuilder._callable_region_args(input)
        else:
            return OpBuilder._callable_region_no_args(input)


_CallableRegionFuncType: TypeAlias = Callable[
    [OpBuilder, tuple[BlockArgument, ...]], None]
