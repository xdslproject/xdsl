from __future__ import annotations

from typing import ParamSpec, Callable, TypeAlias, overload

from dataclasses import dataclass

from xdsl.ir import Operation, OperationInvT, Attribute, Region, Block, BlockArgument
from xdsl.dialects.builtin import FunctionType

_P = ParamSpec('_P')


@dataclass
class Builder:
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
        if isinstance(input, tuple):
            return Builder._callable_region_args(input)
        else:
            return Builder._callable_region_no_args(input)


_CallableRegionFuncType: TypeAlias = Callable[
    [Builder, tuple[BlockArgument, ...]], None]
