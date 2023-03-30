from __future__ import annotations

from typing import Callable, TypeAlias, overload

from dataclasses import dataclass
from xdsl.dialects.builtin import ArrayAttr

from xdsl.ir import OperationInvT, Attribute, Region, Block, BlockArgument


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

    def insert(self, op: OperationInvT) -> OperationInvT:
        """
        Inserts `op` in `self.block` at the end of the block.
        """

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


_CallableRegionFuncType: TypeAlias = Callable[
    [Builder, tuple[BlockArgument, ...]], None]
