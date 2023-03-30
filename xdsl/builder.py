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
    def region(func: Callable[[Builder], None]) -> Region:
        """
        Generates a single-block region.
        """

        block = Block()
        builder = Builder(block)

        func(builder)

        return Region.from_block_list([block])

    @staticmethod
    def callable_region(
        input_types: list[Attribute] | ArrayAttr[Attribute]
    ) -> Callable[[_CallableRegionFuncType], Region]:
        """
        Constructs a single-block region, containing the implementation of a
        function.
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


_CallableRegionFuncType: TypeAlias = Callable[
    [Builder, tuple[BlockArgument, ...]], None]
